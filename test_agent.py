 
from agent import DataScienceExpertAgent

def test_agent():
    """Test the Data Science Expert Agent"""
    print("=" * 70)
    print("🧪 TESTING DATA SCIENCE EXPERT AI AGENT")
    print("=" * 70)
    
    try:
        # Initialize agent
        print("\n1. Initializing agent...")
        agent = DataScienceExpertAgent()
        print("✅ Agent initialized successfully!\n")
        
        # Test 1: Generate questions
        print("=" * 70)
        print("TEST 1: Generating Hard Questions")
        print("=" * 70)
        questions = agent.generate_hard_questions("Machine Learning", num_questions=3)
        print(questions)
        
        # Test 2: Answer a question
        print("\n" + "=" * 70)
        print("TEST 2: Answering a Question")
        print("=" * 70)
        answer = agent.answer_question("Explain the difference between bagging and boosting with examples")
        print(answer)
        
        # Test 3: Quick chat
        print("\n" + "=" * 70)
        print("TEST 3: Quick Chat")
        print("=" * 70)
        response = agent.chat_with_agent("What are the top 3 skills a data scientist should master?")
        print(response)
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_agent()