from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataScienceExpertAgent:
    def __init__(self):
        """Initialize the Data Science Expert AI Agent"""
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        # Initialize client
        self.client = genai.Client(api_key=api_key)
        
        # Expert system prompt
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

        # Chat history
        self.chat_history = []
        
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
    
    def reset_conversation(self):
        """Reset the chat history"""
        self.chat_history = []
        print("Conversation reset successfully!")


def main():
    """Main function to interact with the agent"""
    print("=" * 70)
    print("🤖 DATA SCIENCE EXPERT AI AGENT")
    print("=" * 70)
    print("\nInitializing agent...\n")
    
    try:
        agent = DataScienceExpertAgent()
        print("✅ Agent initialized successfully!\n")
        
        while True:
            print("\n" + "=" * 70)
            print("MENU OPTIONS:")
            print("=" * 70)
            print("1. Generate Hard Questions")
            print("2. Ask a Question")
            print("3. Review Code")
            print("4. Solve a Problem")
            print("5. Chat with Agent")
            print("6. Reset Conversation")
            print("7. Exit")
            print("=" * 70)
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                topic = input("\nEnter topic: ")
                num = input("Number of questions (default 5): ").strip()
                num = int(num) if num else 5
                
                print("\n🔄 Generating questions...\n")
                result = agent.generate_hard_questions(topic, num_questions=num)
                print(result)
                
            elif choice == '2':
                question = input("\nEnter your question: ")
                print("\n🔄 Processing...\n")
                result = agent.answer_question(question)
                print(result)
                
            elif choice == '3':
                print("\nEnter your code (press Enter twice when done):")
                lines = []
                while True:
                    line = input()
                    if line == "" and lines and lines[-1] == "":
                        break
                    lines.append(line)
                code = "\n".join(lines[:-1])
                
                context = input("\nContext (optional): ")
                print("\n🔄 Reviewing code...\n")
                result = agent.review_code(code, context)
                print(result)
                
            elif choice == '4':
                problem = input("\nDescribe your problem: ")
                print("\n🔄 Solving problem...\n")
                result = agent.solve_problem(problem)
                print(result)
                
            elif choice == '5':
                message = input("\nYour message: ")
                print("\n🔄 Processing...\n")
                result = agent.chat_with_agent(message)
                print(result)
                
            elif choice == '6':
                agent.reset_conversation()
                
            elif choice == '7':
                print("\n👋 Thank you for using Data Science Expert AI Agent!")
                break
                
            else:
                print("\n❌ Invalid choice. Please try again.")
                
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nPlease check your API key in .env file")


if __name__ == "__main__":
    main()