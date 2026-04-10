# GitHub Collaboration Strategies

> **Prefer slides?** View the [Git Collaboration Slides](https://kandizzy.github.io/smart-objects-cameras/git-collaboration-slides.html) for a visual walkthrough.

Quick reference for managing student contributions to the smart objects camera template.

---

## Current Workflow (Instructor)

```
Local Machine (Mac) → Git commits → GitHub main branch
         ↓ (scp when ready)
    Raspberry Pis (deploy tested code)
```

**Key points:**
- Develop and test locally when possible
- Commit to main branch (you control stability)
- Deploy to Pis using `scp` for testing
- Main branch stays clean and working

---

## Strategy 1: Student Branches (Recommended)

**Best for:** Small teams, direct instructor oversight, learning Git basics

### Setup:
```bash
# You create branches for each student/team
git checkout -b students/alice
git push -u origin students/alice

git checkout -b students/bob
git push -u origin students/bob

git checkout main  # Return to main
```

### Student Workflow:
```bash
# Student clones repo on their machine or Pi
git clone https://github.com/yourusername/smart-objects-cameras.git
cd smart-objects-cameras
git checkout students/alice

# Make changes
git add person_detector.py
git commit -m "Add zone detection feature"
git push origin students/alice
```

### Review & Merge:
```bash
# You review student branch
git checkout students/alice
git pull

# If good, merge to main
git checkout main
git merge students/alice
git push origin main
```

**Pros:**
- ✅ Simple for students
- ✅ Easy to track individual work
- ✅ You control what goes to main
- ✅ Can work directly on Pis with VS Code Remote

**Cons:**
- ❌ Students can see each other's work
- ❌ Requires Git knowledge

---

## Strategy 2: Fork Model

**Best for:** Larger teams, students building completely different projects, learning OSS workflow

### Setup:
1. Students click "Fork" on GitHub
2. Each student has their own copy: `github.com/alice/smart-objects-cameras`
3. Students work on their fork

### Student Workflow:
```bash
# Student clones their fork
git clone https://github.com/alice/smart-objects-cameras.git

# Make changes on their fork
git add .
git commit -m "My custom feature"
git push origin main  # Pushes to THEIR fork

# Submit Pull Request to your repo
# (via GitHub UI)
```

### Review & Merge:
- Students submit PR from their fork to your main repo
- You review on GitHub
- Click "Merge Pull Request" if approved

**Pros:**
- ✅ Complete independence for students
- ✅ Professional workflow (real OSS experience)
- ✅ Students learn GitHub PR process
- ✅ Can't accidentally break your main repo

**Cons:**
- ❌ More complex for beginners
- ❌ Extra step (fork management)

---

## Strategy 3: Direct Collaboration (Simple)

**Best for:** Quick prototyping, students working directly on Pis, minimal Git usage

### Setup:
```bash
# Students work directly on Pi (no Git)
# Each student makes their own copy of files
cd ~/oak-projects
cp person_detector.py person_detector_alice.py
```

### When Ready to Share:
```bash
# You pull their work manually
scp orbit:~/oak-projects/person_detector_alice.py ./students/

# Commit to their branch
git checkout -b students/alice
git add students/person_detector_alice.py
git commit -m "Alice: Zone detection feature"
git push origin students/alice
```

**Pros:**
- ✅ No Git knowledge required for students
- ✅ Focus on coding, not version control
- ✅ Works great with VS Code Remote SSH
- ✅ Students experiment freely on Pi

**Cons:**
- ❌ Manual work for you to commit
- ❌ Students don't learn Git
- ❌ No version history during development

---

## Quick Decision Guide

**Choose Student Branches if:**
- Students understand basic Git
- You want to review code before merging
- Working on shared template collaboratively

**Choose Fork Model if:**
- Teaching professional workflow
- Students building independent projects
- Want complete isolation between students

**Choose Direct Collaboration if:**
- Students are new to Git
- Focus is on camera/CV learning, not version control
- Quick iteration is priority
- Can teach Git later

---

## Hybrid Approach (Recommended for Classrooms)

**Week 1-4:** Direct collaboration (no Git for students)
- Students work on Pis, make personal copies
- Focus on learning camera/detection/Discord
- You commit interesting work to student branches

**Week 5+:** Introduce branching
- Students learn basic Git commands
- Work on their own branches
- Submit work via push to their branch

**Advanced students:** Can use fork model for final projects

---

## Important: What NOT to Commit

Create `.gitignore`:
```
# Secrets
.env
*.env

# Generated files
camera_status.json
latest_frame.jpg
*.log

# Python
__pycache__/
*.pyc
venv/
.venv/

# OS
.DS_Store
```

**Never commit:**
- Discord bot tokens
- Webhook URLs
- SSH keys
- Personal .env files

---

## Quick Commands Reference

### For You (Instructor):
```bash
# Create student branch
git checkout -b students/alice
git push -u origin students/alice

# Review student work
git checkout students/alice
git pull

# Merge to main
git checkout main
git merge students/alice
git push origin main

# Deploy to Pi
scp person_detector.py orbit:~/oak-projects/
```

### For Students (Branch Model):
```bash
# Get latest code
git checkout students/alice
git pull

# Make changes
git add .
git commit -m "Description of changes"
git push origin students/alice
```

### For Students (Fork Model):
```bash
# Keep fork updated with your main repo
git remote add upstream https://github.com/yourusername/smart-objects-cameras.git
git fetch upstream
git merge upstream/main
```

---

## Tips for Class Discussion

**Explain to students:**
1. Main branch = stable, working template
2. Your branch = your experiments (can break things!)
3. Commits = save points (describe what you did)
4. Push = share your work with others
5. Pull = get latest code

**First day:**
- Show how to clone
- Show how to make their branch
- Show commit → push workflow
- That's it! Keep it simple.

**As they advance:**
- Introduce pull requests
- Teach merge conflict resolution
- Show fork model for independent projects

---

## Resources

- **GitHub Basics:** https://docs.github.com/en/get-started
- **Git Cheat Sheet:** https://education.github.com/git-cheat-sheet-education.pdf
- **VS Code Git:** Built-in, super easy - just click the Git icon!

---

*Discuss with your team which strategy fits your course goals and student skill levels.*
