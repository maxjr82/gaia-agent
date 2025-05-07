# 🤖 GAIA Agent – Hugging Face AI Agents Course Final Project

Welcome to the final assignment of the Hugging Face AI Agents Course! This
repository contains a custom-built AI agent evaluated on the GAIA benchmark —
a rigorous test designed to challenge AI assistants on real-world, multi-step
tasks.

## 🌍 Introduction to the GAIA Benchmark

GAIA is a comprehensive benchmark of **466 real-world, multi-step factoid questions**
designed to evaluate an AI assistant’s ability to reason, browse the web,
handle multimodal inputs, and leverage external tools in zero-shot settings.
Unlike traditional benchmarks, GAIA tasks remain conceptually simple for humans
(≈92 % human accuracy) yet challenge even state-of-the-art models augmented
with plugins (≈15 % accuracy for GPT-4).

#### GAIA’s Core Principles
- **Real-world difficulty**: Requires multi-hop retrieval, web scraping, PDF/image parsing, etc.  
- **Human interpretability**: Tasks are easy to validate via concise, unambiguous answers.  
- **Non-gameability**: Brute forcing is ineffective—complete tool chains are needed.  
- **Simplicity of evaluation**: Unique factoid answers allow for automatic scoring.

---

### 🎓 Course Context: HuggingFace AI Agents Final Project

This agent is the **capstone** of the HuggingFace AI Agents course’s final unit.  
- **Goal**: Build an agent that uses multiple tools to answer GAIA questions zero-shot.  
- **Certification**: Score **≥ 30 %** (i.e., **6/20** correct) on a held-out set of 20 GAIA questions to earn your course certificate.  
- **Leaderboard**: Submit your score to the student leaderboard to compare with peers.

> **Heads Up**: This practical unit demands advanced coding and problem-solving with minimal hand-holding.

---

## 🚀 Getting Started: Local Installation & Testing

1. **Clone the Repository**  

   ```bash
   git clone https://github.com/maxjr82/gaia-agent.git
   cd gaia-agent
   ```

2. **Create a virtual env & install**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure API keys**

   OPENROUTER_API_KEY=your_openrouter_key
   WOLFRAM_APP_ID=your_wolframalpha_app_id  


## 📥 Accessing GAIA Questions
You can access a subset of GAIA questions via the Hugging Face API:

https://agents-course-unit4-scoring.hf.space/questions

⚠️ Note: This link is temporary and is expected to expire by the end of May 2025.