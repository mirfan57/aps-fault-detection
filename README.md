# neurolab-mongo-python

![image](https://user-images.githubusercontent.com/57321948/196933065-4b16c235-f3b9-4391-9cfe-4affcec87c35.png)

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
