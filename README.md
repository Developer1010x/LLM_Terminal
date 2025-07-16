# LLM_Terminal

# 🧠 LLM_Terminal

**LLM_Terminal** is a smart terminal assistant that uses a Large Language Model (LLM) to help you:

- 🧾 Understand cryptic terminal error messages
- 🛠️ Know the correct shell command for your task
- ⚡ Learn CLI tools on the fly, without Googling

Perfect for beginners, learners, or anyone wanting a faster way to work with the command line.

---

## 🖥️ How It Works

You type a question or paste an error into the terminal.  
The tool sends it to an LLM (like OpenAI's GPT) and gives you back a plain-language answer or a recommended command.

Example:
```bash
> I want to unzip a .tar.gz file
Try: tar -xvzf filename.tar.gz
> zsh: command not found: curl
This means `curl` is not installed. Try: sudo apt install curl
```
LLM_Terminal/
├── app.py        # CLI interface and input/output handling
├── llm.py        # Interacts with LLM and formats the prompts
├── LICENSE       # License file (Unlicense)
├── README.md     # You're reading it!
