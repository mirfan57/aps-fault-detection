### Step 1 - Install the requirements

```bash
pip install -r requirements.txt
```

### Step 2 - Run main.py file

```bash
python main.py
```

Please follow these commands

While starting a new project we can initialize git by
```
git init
```

We can even clone an existing github repo

```
git clone <github_url>
```
Note: Clone/Download repo in local system

Add changes made to file in git staging area by
```
git add <filename>
```
Note: We use "filename" to add a specific file 
or
```
git add .
```
Note: We use "." to add everything to staging area

Create commit
```
git commit -m "your_message"
```

Push to remote origin main branch
```
git push origin main
```
Note: Origin contains url to your github repo and main is your branch name

To push changes forcefully
```
git push origin main -f
```

To pull changes from github repo
```
git pull origin main
```
Note: To solve the merge issues

For commit issues
First check all commits done using 
```
git log
```
To move the HEAD pointer reference to a given commit msg having commit_ID without deleting the earlier commit contents.

`git reset` is used to undo local changes to the state of a Git repo
```
git reset --soft <first_4_words_of_commit_ID>
```
Now follow same steps from `git add .`
