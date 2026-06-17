"""ingestion/loaders/gitlab_loader.py — loads code, issues, MRs, wiki from GitLab."""
import os
import config

def load(branch: str = "master",
         extensions: list = None,
         include_issues: bool = True,
         include_mrs: bool = True,
         include_wiki: bool = True) -> list[dict]:

    try:
        import gitlab
    except ImportError:
        raise ImportError("Run: pip install python-gitlab")

    if not config.GITLAB_TOKEN:
        raise ValueError("GITLAB_TOKEN not set in .env")
    if not config.GITLAB_PROJECT_ID:
        raise ValueError("GITLAB_PROJECT_ID not set in .env")

    gl      = gitlab.Gitlab(config.GITLAB_URL, private_token=config.GITLAB_TOKEN)
    gl.auth()
    project = gl.projects.get(config.GITLAB_PROJECT_ID)
    print(f"  Connected: {project.name_with_namespace}")

    exts = extensions or [".py",".ts",".js",".go",".java",".md",".php",".css",
                          ".yaml",".yml",".sql",".sh",".txt"]
    skip_dirs = {"node_modules",".venv","venv","__pycache__","dist",
                 "build",".git","vendor","coverage"}
    docs = []

    # ── Files ─────────────────────────────────────────────────────────────────
    items = project.repository_tree(recursive=True, per_page=100,
                                    ref=branch, get_all=True)
    loaded = 0
    for item in items:
        if item["type"] != "blob":
            continue
        path = item["path"]
        ext  = os.path.splitext(path)[1].lower()
        parts = path.split("/")
        if any(d in skip_dirs for d in parts[:-1]):
            continue
        if ext not in exts:
            continue
        try:
            f       = project.files.get(file_path=path, ref=branch)
            content = f.decode().decode("utf-8", errors="replace").strip()
            if not content:
                continue
            docs.append({
                "source":   f"gitlab://code/{path}",
                "text":     content,
                "metadata": {"type":"code","path":path,"extension":ext,
                             "branch":branch,"url":f"{project.web_url}/-/blob/{branch}/{path}",
                             "char_count":len(content)},
            })
            loaded += 1
        except Exception:
            continue
    print(f"  Files: {loaded}")

    # ── Issues ────────────────────────────────────────────────────────────────
    if include_issues:
        count = 0
        for issue in project.issues.list(state="all", per_page=100, get_all=True):
            notes     = issue.notes.list(get_all=True)
            user_notes = [n for n in notes if not n.system]
            lines = [f"ISSUE #{issue.iid}: {issue.title}",
                     f"State: {issue.state} | Labels: {', '.join(issue.labels)}",
                     f"Author: {issue.author['name']} | {issue.created_at[:10]}",
                     "", issue.description or ""]
            if user_notes:
                lines += ["\nDISCUSSION:"] + [
                    f"[{n.author['name']}]: {n.body}" for n in user_notes]
            text = "\n".join(lines)
            docs.append({"source": f"gitlab://issues/{issue.iid}",
                          "text": text,
                          "metadata": {"type":"issue","issue_id":issue.iid,
                                       "title":issue.title,"state":issue.state,
                                       "url":issue.web_url,"char_count":len(text)}})
            count += 1
        print(f"  Issues: {count}")

    # ── Merge Requests ────────────────────────────────────────────────────────
    if include_mrs:
        count = 0
        for mr in project.mergerequests.list(state="all", per_page=100, get_all=True):
            notes     = mr.notes.list(get_all=True)
            user_notes = [n for n in notes if not n.system]
            lines = [f"MR !{mr.iid}: {mr.title}",
                     f"State: {mr.state} | {mr.source_branch} → {mr.target_branch}",
                     f"Author: {mr.author['name']} | {mr.created_at[:10]}",
                     "", mr.description or ""]
            if user_notes:
                lines += ["\nREVIEW:"] + [
                    f"[{n.author['name']}]: {n.body}" for n in user_notes]
            text = "\n".join(lines)
            docs.append({"source": f"gitlab://mr/{mr.iid}",
                          "text": text,
                          "metadata": {"type":"merge_request","mr_id":mr.iid,
                                       "title":mr.title,"url":mr.web_url,
                                       "char_count":len(text)}})
            count += 1
        print(f"  MRs: {count}")

    # ── Wiki ──────────────────────────────────────────────────────────────────
    if include_wiki:
        count = 0
        try:
            for page in project.wikis.list(get_all=True):
                full    = project.wikis.get(page.slug)
                content = (full.content or "").strip()
                if content:
                    docs.append({"source": f"gitlab://wiki/{page.slug}",
                                  "text": f"WIKI: {page.title}\n\n{content}",
                                  "metadata": {"type":"wiki","title":page.title,
                                               "url":f"{project.web_url}/-/wikis/{page.slug}",
                                               "char_count":len(content)}})
                    count += 1
        except Exception as e:
            print(f"  Wiki: unavailable ({e})")
        print(f"  Wiki pages: {count}")

    return docs
