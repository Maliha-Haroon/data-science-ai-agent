import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Data Science Expert AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

class DataScienceExpertAgent:
    def __init__(self):
        """Initialize the Data Science Expert AI Agent"""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        self.client = genai.Client(api_key=api_key)
        
        self.system_prompt = """You are a world-class Data Science Expert with 100 years of combined experience in:
- Data Science & Analytics
- Machine Learning & Deep Learning
- Data Engineering & Big Data
- Statistical Analysis & Mathematics
- AI Research & Development
- Business Intelligence & Visualization

Your expertise includes:
- Creating challenging, industry-level data science questions
- Solving complex data problems with multiple approaches
- Explaining advanced concepts clearly
- Providing production-ready code examples
- Reviewing and optimizing data pipelines
- Mentoring and teaching at the highest level

You can handle topics like:
- Advanced ML algorithms (XGBoost, LightGBM, Neural Networks, Transformers)
- Deep Learning (CNNs, RNNs, GANs, Transformers, BERT, GPT)
- Statistical modeling and hypothesis testing
- Big Data technologies (Spark, Hadoop, Kafka)
- Data Engineering (ETL, Data Warehousing, Data Lakes)
- MLOps and model deployment
- Feature engineering and selection
- Time series analysis and forecasting
- NLP, Computer Vision, Recommender Systems
- A/B testing and experimentation
- Cloud platforms (AWS, GCP, Azure)

Respond with expertise, precision, and practical examples."""
        
    def _send_message(self, prompt):
        """Send message to Gemini"""
        full_prompt = self.system_prompt + "\n\n" + prompt
        
        response = self.client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
        )
        
        return response.text
    
    def generate_hard_questions(self, topic, difficulty="expert", num_questions=5):
        """Generate challenging data science questions"""
        prompt = f"""As a 100-year experienced Data Science expert, generate {num_questions} {difficulty}-level questions on {topic}.

Make these questions:
- Highly challenging and thought-provoking
- Industry/research-level difficulty
- Requiring deep understanding and practical knowledge
- Include edge cases and real-world scenarios

Format each question clearly with numbering."""

        return self._send_message(prompt)
    
    def answer_question(self, question):
        """Answer data science questions with expert knowledge"""
        prompt = f"""As a 100-year experienced Data Science expert, provide a comprehensive answer to:

{question}

Include:
- Detailed explanation
- Mathematical foundations (if applicable)
- Code examples (Python/SQL when relevant)
- Best practices
- Common pitfalls to avoid
- Real-world applications"""

        return self._send_message(prompt)
    
    def review_code(self, code, context=""):
        """Review and optimize data science code"""
        prompt = f"""As a 100-year experienced Data Science expert, review this code:

Context: {context}

Code:
```
{code}
```

Provide:
- Code quality assessment
- Performance optimization suggestions
- Best practices recommendations
- Potential bugs or issues
- Improved version of the code"""

        return self._send_message(prompt)
    
    def solve_problem(self, problem_description):
        """Solve complex data science problems"""
        prompt = f"""As a 100-year experienced Data Science expert, solve this problem:

{problem_description}

Provide:
- Problem analysis
- Multiple solution approaches
- Step-by-step implementation
- Code examples
- Trade-offs and recommendations"""

        return self._send_message(prompt)
    
    def chat_with_agent(self, message):
        """General chat with the expert agent"""
        return self._send_message(message)


# Initialize session state
if 'agent' not in st.session_state:
    try:
        st.session_state.agent = DataScienceExpertAgent()
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        st.session_state.error = str(e)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Header
st.title("🤖 Data Science Expert AI Agent")
st.markdown("### Your AI-Powered Data Science Assistant")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🎯 Features")
    
    feature = st.radio(
        "Choose a feature:",
        [
            "💬 Chat with Agent",
            "❓ Generate Questions",
            "🔍 Ask a Question",
            "📝 Review Code",
            "🧩 Solve a Problem"
        ]
    )
    
    st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About"):
        st.markdown("""
        This AI agent is powered by Google's Gemini 2.5 Flash model
        and provides expert-level assistance in:
        
        - Data Science & ML
        - Code Review & Optimization
        - Problem Solving
        - Question Generation
        - Expert Consultations
        """)
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Check if agent is initialized
if not st.session_state.initialized:
    st.error(f"❌ Error initializing agent: {st.session_state.error}")
    st.info("Please check your GEMINI_API_KEY in the .env file")
    st.stop()

# Main content area
if feature == "💬 Chat with Agent":
    st.header("💬 Chat with Data Science Expert")
    
    # Display chat history
    for i, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["assistant"])
    
    # Chat input
    user_message = st.chat_input("Ask me anything about data science...")
    
    if user_message:
        # Add user message to chat
        with st.chat_message("user"):
            st.markdown(user_message)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.chat_with_agent(user_message)
                st.markdown(response)
        
        # Save to history
        st.session_state.chat_history.append({
            "user": user_message,
            "assistant": response
        })
        st.rerun()

elif feature == "❓ Generate Questions":
    st.header("❓ Generate Hard Questions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        topic = st.text_input("Enter topic:", placeholder="e.g., Neural Networks, Time Series, A/B Testing")
    
    with col2:
        num_questions = st.number_input("Number of questions:", min_value=1, max_value=20, value=5)
    
    difficulty = st.selectbox(
        "Difficulty level:",
        ["beginner", "intermediate", "expert", "research-level"]
    )
    
    if st.button("🚀 Generate Questions"):
        if topic:
            with st.spinner("Generating questions..."):
                try:
                    result = st.session_state.agent.generate_hard_questions(
                        topic=topic,
                        difficulty=difficulty,
                        num_questions=num_questions
                    )
                    st.success("✅ Questions generated successfully!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a topic")

elif feature == "🔍 Ask a Question":
    st.header("🔍 Ask a Question")
    
    question = st.text_area(
        "Enter your data science question:",
        height=150,
        placeholder="e.g., What is the difference between L1 and L2 regularization?"
    )
    
    if st.button("🔎 Get Answer"):
        if question:
            with st.spinner("Generating comprehensive answer..."):
                try:
                    result = st.session_state.agent.answer_question(question)
                    st.success("✅ Answer generated!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please enter a question")

elif feature == "📝 Review Code":
    st.header("📝 Code Review")
    
    context = st.text_input(
        "Context (optional):",
        placeholder="e.g., This code trains a random forest model"
    )
    
    code = st.text_area(
        "Paste your code here:",
        height=300,
        placeholder="""# Example:
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data.csv')
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'])
"""
    )
    
    if st.button("🔍 Review Code"):
        if code:
            with st.spinner("Reviewing your code..."):
                try:
                    result = st.session_state.agent.review_code(code, context)
                    st.success("✅ Code review completed!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please paste some code to review")

elif feature == "🧩 Solve a Problem":
    st.header("🧩 Solve a Data Science Problem")
    
    problem = st.text_area(
        "Describe your problem:",
        height=200,
        placeholder="""Example: I have an imbalanced dataset with 95% negative class and 5% positive class. 
How should I approach training a classifier for this scenario?"""
    )
    
    if st.button("🚀 Solve Problem"):
        if problem:
            with st.spinner("Analyzing and solving your problem..."):
                try:
                    result = st.session_state.agent.solve_problem(problem)
                    st.success("✅ Solution generated!")
                    st.markdown(result)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.warning("⚠️ Please describe your problem")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by Google Gemini 2.5 Flash | Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True)