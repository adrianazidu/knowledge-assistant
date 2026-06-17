"""ingestion/loaders/gitlab_loader.py — loads code, issues, MRs, wiki from GitLab."""
import os
import config
import io
import tarfile

class GitlabLoader:

    def __init__(self,
                branch: str = "master",
                extensions: list = None,
                include_issues: bool = True,
                include_mrs: bool = True,
                include_wiki: bool = True):

        try:
            import gitlab
        except ImportError:
            raise ImportError("Run: pip install python-gitlab")

        if not config.GITLAB_TOKEN:
            raise ValueError("GITLAB_TOKEN not set in .env")
        if not config.GITLAB_PROJECT_ID:
            raise ValueError("GITLAB_PROJECT_ID not set in .env")

        self.branch = branch
        self.extensions = extensions
        self.include_issues = include_issues
        self.include_mrs = include_mrs
        self.include_wiki = include_wiki

        gl      = gitlab.Gitlab(config.GITLAB_URL, private_token=config.GITLAB_TOKEN)
        gl.auth()

        self.project = gl.projects.get(config.GITLAB_PROJECT_ID)
        print(f"  Connected: {self.project.name_with_namespace} branch {self.branch}")

    

    def load(self)-> list[dict]:
       
        self.docs = []

        self.load_files()

        if self.include_issues:
            self.load_issues()

        if self.include_mrs:
            self.load_mergereq()

        
        if self.include_wiki:
            self.load_wiki()


        return self.docs


    def load_from_archive(self)->list[dict]:
        
        # 1. download all branch files in tar archive
        print("Download dell'intero repository in corso...")
        archive_bytes = project.repository_archive(sha=branch)

        # 2. extract and process in memory
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
            for member in tar.getmembers():
                # check is file (blob) and not directory
                if not member.isfile():
                    continue
                    
                path = member.name
                ext = os.path.splitext(path)[1].lower()
                parts = path.split("/")
                
                # apply filters
                if any(d in skip_dirs for d in parts[:-1]):
                    continue
                if ext not in exts:
                    continue
                    
                # extract file
                f = tar.extractfile(member)
                if f is not None:
                    content = f.read().decode("utf-8", errors="replace").strip()
                    if not content:
                        continue
                    self.docs.append({
                    "source":   f"local/tar/{path}",
                    "text":     content,
                    "metadata": {"type":"code","path":path,"extension":ext,
                                "branch":self.branch,"url":f"{self.project.web_url}/-/blob/{self.branch}/{path}",
                                "char_count":len(content)},
                    })
                        
                #embed and ave


    def load_files(self)->list[dict]:

        #define ext if extensions not defined
        exts = self.extensions or [".py",".ts",".js",".go",".java",".md",".php",".css",
                            ".yaml",".yml",".sql",".sh",".txt"]

        skip_dirs = {"node_modules",".venv","venv","__pycache__","dist",
                    "build",".git","vendor","coverage","updates","tests","css","fonts"}
   
        IGNORE_SUBFOLDERS = ["src/libs", "public/libs","onboard"]

        # ── Files ─────────────────────────────────────────────────────────────────
        items = self.project.repository_tree(recursive=True, per_page=2000,
                                                ref=self.branch, get_all=False)
        loaded = 0
        for item in items:
            print(f" analyze item {item['path']}", flush=True )

            if item["type"] != "blob":
                continue
            path = item["path"]
            ext  = os.path.splitext(path)[1].lower()
            parts = path.split("/")

            #ignore the ones in skip dirs
            #if at least one (ANY) of the parent dirs of the file is in skipped list
            if any(d in skip_dirs for d in parts[:-1]):
                print("path present in skip dirs, ignoring..", flush=True)
                continue

            if any (path.startswith(folder + "/") for folder in IGNORE_SUBFOLDERS):
                print("path present in ignore subfolders, ignoring...", flush=True)
                continue
    
            #ignore the not enabled extensions
            if ext not in exts:
                print(f"extension {ext} not considered, ignoring..", flush=True)
                continue
            try:
                f       = self.project.files.get(file_path=path, ref=self.branch)
                content = f.decode().decode("utf-8", errors="replace").strip()
                #first f.decode() is a method of gitlab - converts file in a bytes
                #the second decode converts file in a string
            
            #if the file is empty
                if not content:
                    continue
                self.docs.append({
                    "source":   f"gitlab://code/{path}",
                    "text":     content,
                    "metadata": {"type":"code","path":path,"extension":ext,
                                "branch":self.branch,"url":f"{self.project.web_url}/-/blob/{self.branch}/{path}",
                                "char_count":len(content)},
                })
                loaded += 1
            except Exception:
                continue
        print(f"  Files: {loaded}")

    def load_issues(self)->list[dict]:

        count = 0
        for issue in self.project.issues.list(state="closed", per_page=2, get_all=False):
            print(f" Analyze issue {issue.title}") 
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
            self.docs.append({"source": f"gitlab://issues/{issue.iid}",
                          "text": text,
                          "metadata": {"type":"issue","issue_id":issue.iid,
                                       "title":issue.title,"state":issue.state,
                                       "url":issue.web_url,"char_count":len(text)}})
            count += 1
        print(f"  Issues: {count}")

    def load_mergereq(self)->list[dict]:
   
        count = 0
        #state all/merged
        for mr in self.project.mergerequests.list(state="merged", per_page=2, get_all=False):
            print(f"Analyze merge request {mr.title}")
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
            self.docs.append({"source": f"gitlab://mr/{mr.iid}",
                            "text": text,
                            "metadata": {"type":"merge_request","mr_id":mr.iid,
                                        "title":mr.title,"url":mr.web_url,
                                        "char_count":len(text)}})
            count += 1
        print(f"  MRs: {count}")

    def load_wiki(self)->list[dict]:
   
        count = 0
        try:
            for page in self.project.wikis.list(get_all=False, per_page=3):
                full    = self.project.wikis.get(page.slug)
                content = (full.content or "").strip()
                if content:
                    self.docs.append({"source": f"gitlab://wiki/{page.slug}",
                                  "text": f"WIKI: {page.title}\n\n{content}",
                                  "metadata": {"type":"wiki","title":page.title,
                                               "url":f"{self.project.web_url}/-/wikis/{page.slug}",
                                               "char_count":len(content)}})
                    count += 1
        except Exception as e:
            print(f"  Wiki: unavailable ({e})")
        print(f"  Wiki pages: {count}")


