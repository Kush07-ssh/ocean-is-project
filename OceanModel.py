import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
# Function to generate the report
def llm_analysis(scores):
    """
    Generates a personality report using Google Gemini 1.5 Flash.

    Args:
        scores (dict): Dictionary of OCEAN scores (e.g., {'Openness': 35, ...})
        api_key (str): Google API Key

    Returns:
        str: Markdown formatted report.
    """

    # 1. Initialize Gemini Model (Using 1.5 Flash for speed)
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7
    )

    # 2. Define the Prompt Template
    template = """
        You are an objective psychometrician and career counselor. 
        You are analyzing Big Five (OCEAN) personality scores to create a profile.

        **User's Normalized Scores (0-100 Scale):**
        {scores_context}

        *(Note: 0 is the absolute minimum, 50 is average, 100 is the absolute maximum)*

        **Instructions:**
        1. **Contextualize:** Interpret the scores knowing they are normalized percentages. 
        2. **Be Honest & Direct:** Provide a balanced, realistic view. Do not sugarcoat or rely on toxic positivity. If a score indicates a tendency to be disorganized, easily stressed, uncooperative, or withdrawn, state it plainly. 
        3. **Keep it Simple:** Use clear, accessible, everyday language. Avoid dense academic jargon.

        **Structure the report EXACTLY as follows in clean Markdown:**

        - **Executive Summary:** A concise, 2-sentence overview of their core personality type.
        - **Key Strengths:** 3 brief bullet points highlighting their clearest advantages based on their highest relative scores.
        - **Genuine Blind Spots:** 2-3 specific areas they will likely struggle with. Be constructive but completely honest.
        - **Work & Career Style:** 2-3 sentences explaining how they naturally operate in a professional environment and team setting.
        - **Actionable Growth Advice:** 2-3 practical, realistic steps they can take immediately to mitigate their biggest blind spot.
        """

    prompt = PromptTemplate(
        input_variables=["scores_context"],
        template=template
    )

    # 3. Create the Chain
    chain = prompt | llm | StrOutputParser()

    # 4. Format scores for the prompt
    scores_text = "\n".join([f"- {trait}: {score}" for trait, score in scores.items()])
    print(scores_text)
    # 5. Run the Chain
    try:
        response = chain.invoke({"scores_context": scores_text})
        print(response)
        return response
    except Exception as e:
        return f"Error generating report: {str(e)}"
