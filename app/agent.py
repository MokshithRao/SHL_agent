import os
import json
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

from app.models import ChatMessage, ChatResponse, RecommendationItem
from app.retriever import AssessmentRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SHLConversationalAgent:
    def __init__(self):
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment variables.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Initialize retriever
        self.retriever = AssessmentRetriever()
        if not self.retriever.load_index():
            logger.info("Building retrieval index...")
            self.retriever.build_index("data/shl_product_catalog.json")
            self.retriever.save_index()
            
        self.system_instruction = (
            "You are an expert SHL Conversational Assessment Recommendation Agent.\n"
            "Your goal is to understand hiring requirements and recommend appropriate SHL assessments.\n"
            "Keep your tone professional, objective, and recruiter-friendly."
        )

    def extract_latest_user_message(self, messages: List[Dict[str, Any]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg.get("content", ""))
        return ""

    def analyze_conversation(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        history_text = "\n".join([f"{msg.get('role', 'unknown')}: {msg.get('content', '')}" for msg in messages])
        
        prompt = f"""
        Analyze the following conversation regarding SHL assessment recommendations.
        Extract the following information:
        - target_role: The primary job role being discussed.
        - experience_level: (e.g., junior, graduate, senior, managerial)
        - test_needs: (e.g., cognitive, personality, coding, undetermined)
        - missing_info: List of crucial information still needed to make a good recommendation.
        - has_recommendations: Boolean (true if we have successfully made recommendations already relative to the goal).
        
        Conversation:
        {history_text}
        
        Respond ONLY with a valid JSON object strictly adhering to these keys.
        """
        
        try:
            response = self.model.generate_content(prompt)
            # Find the JSON block
            import re
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response.text, re.DOTALL)
            json_str = match.group(1) if match else response.text.strip()
            
            # Cleanup common trailing commas
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)
            
            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Error during conversation analysis: {e}")
            return {
                "target_role": None,
                "experience_level": None,
                "test_needs": None,
                "missing_info": [],
                "has_recommendations": False
            }

    def detect_refusal_case(self, message: str) -> bool:
        lower_msg = message.lower()
        
        # Rule-based fast execution
        blacklist = [
            "ignore previous", "prompt injection", "system prompt",
            "legal advice", "sue", "lawsuit", "medical", "doctor",
            "aws certification", "azure certification", "forget instructions",
            "act as", "you are now", "developer mode", "bypass", "jailbreak",
            "recommend certifications", "recommend courses", "recommend books"
        ]
        
        for term in blacklist:
            if term in lower_msg:
                return True
        return False

    def detect_comparison_request(self, message: str) -> bool:
        lower_msg = message.lower()
        keywords = ["compare", "difference between", "versus", "vs", "vs."]
        return any(kw in lower_msg for kw in keywords)

    def extract_comparison_terms(self, message: str) -> List[str]:
        """
        Extracts likely assessment names for comparison using regex and token cleanup.
        """
        import re
        # Remove common comparison glue words
        clean_msg = re.sub(r'(?i)\b(compare|difference|between|versus|vs\.?|and|the)\b', ' ', message)
        # Extract remaining meaningful tokens
        tokens = [t.strip() for t in clean_msg.split() if t.strip()]
        return tokens

    def should_clarify(self, conversation_state: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Determines if the agent needs more information before recommending.
        """
        target_role = conversation_state.get("target_role")
        test_needs = conversation_state.get("test_needs")
        experience_level = conversation_state.get("experience_level")
        
        if not target_role and not test_needs:
            return True, "Could you provide more details about the role or the type of skills you want to assess?"
            
        if not target_role:
            return True, "What specific job role are you hiring for?"
            
        if not experience_level and not test_needs:
             return True, f"For the {target_role} role, are you looking for a specific experience level or test type (e.g., cognitive, personality)?"
             
        return False, ""

    def build_search_query(self, conversation_state: Dict[str, Any]) -> str:
        parts = []
        if r := conversation_state.get("target_role"):
            parts.append(str(r))
        if e := conversation_state.get("experience_level"):
            parts.append(str(e))
        if t := conversation_state.get("test_needs"):
            parts.append(str(t))
            
        return " ".join(parts) if parts else "general assessment"

    def generate_recommendations(self, results: List[Dict[str, Any]]) -> List[RecommendationItem]:
        """
        Converts raw retrieval results into RecommendationItem objects.
        """
        recommendations = []
        seen_names = set()
        
        for res in results:
            ass = res["assessment"]
            if ass.name in seen_names:
                continue
            seen_names.add(ass.name)
            
            # Infer test type from keys/name
            test_type = "Assessment"
            combined_text = (ass.name + " " + " ".join(ass.keys or [])).lower()
            if "cognitive" in combined_text or "ability" in combined_text:
                test_type = "Cognitive"
            elif "personality" in combined_text or "behavior" in combined_text:
                test_type = "Personality"
            elif "skill" in combined_text or "code" in combined_text or "coding" in combined_text:
                test_type = "Skills"
                
            recommendations.append(RecommendationItem(
                name=ass.name,
                url=ass.link or "",
                test_type=test_type
            ))
            
        return recommendations

    def generate_comparison_response(self, message: str, assessments: List[Dict[str, Any]]) -> str:
        if not assessments:
            return "I don't have enough assessment information gathered to compare them properly."
            
        context_docs = "\n\n".join([f"Name: {res['assessment'].name}\nDescription: {res['assessment'].description}" for res in assessments])
        
        prompt = f"""
        {self.system_instruction}
        
        The user wants to compare assessments.
        User request: {message}
        
        Here is the catalog information for the retrieved assessments:
        {context_docs}
        
        Provide a concise comparison of these assessments based ONLY on the data provided above. Do not invent details.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during comparison generation: {e}")
            return "I encountered an issue generating a comparison. Please provide more details or try again."

    def generate_reply(self, message: str, assessments: List[Dict[str, Any]], conversation_state: Dict[str, Any]) -> str:
        context_docs = "\n\n".join([f"Name: {res['assessment'].name}\nDescription: {res['assessment'].description}" for res in assessments])
        
        prompt = f"""
        {self.system_instruction}
        
        User Context: {json.dumps(conversation_state)}
        User Message: {message}
        
        Retrieved Candidate Assessments:
        {context_docs}
        
        Generate a concise, professional reply recommending the best assessments based on the user's needs.
        Base your recommendations ONLY on the retrieved assessments provided above.
        Only mention assessment names explicitly provided in the retrieved assessment list.
        Do not invent or infer new assessment names or features.
        Do not output giant paragraphs; be brief.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error during response generation: {e}")
            return "I encountered an issue generating a response. Please provide more details or try again."

    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        latest_msg = self.extract_latest_user_message(messages)
        if not latest_msg:
             return {
                "reply": "I didn't catch that. Could you please provide more details?",
                "recommendations": [],
                "end_of_conversation": False
            }
            
        if self.detect_refusal_case(latest_msg):
            return {
                "reply": "I can only assist with SHL assessment recommendations and related queries. I cannot provide legal, medical advice, or answer unrelated questions.",
                "recommendations": [],
                "end_of_conversation": False
            }
            
        conv_state = self.analyze_conversation(messages)
        
        is_comparison = self.detect_comparison_request(latest_msg)
        
        needs_clarification, clar_question = self.should_clarify(conv_state)
        
        if needs_clarification and not is_comparison:
            return {
                "reply": clar_question,
                "recommendations": [],
                "end_of_conversation": False
            }
            
        search_query = self.build_search_query(conv_state)
        if is_comparison:
            comp_terms = self.extract_comparison_terms(latest_msg)
            search_query += " " + " ".join(comp_terms)
            
        raw_results = self.retriever.search(search_query, top_k=5)
        
        if not raw_results:
            return {
                "reply": "I couldn't find suitable SHL assessments for that request. Could you provide more details about the role, skills, or experience level required?",
                "recommendations": [],
                "end_of_conversation": False
            }
            
        recommendations = self.generate_recommendations(raw_results)
        
        if is_comparison:
            reply = self.generate_comparison_response(latest_msg, raw_results)
        else:
            reply = self.generate_reply(latest_msg, raw_results, conv_state)
            
        return {
            "reply": reply,
            "recommendations": [rec.model_dump() for rec in recommendations],
            "end_of_conversation": False
        }

if __name__ == "__main__":
    agent = SHLConversationalAgent()
    
    print("\n--- Test 1: Vague Query ---")
    msg1 = [{"role": "user", "content": "I need an assessment."}]
    res1 = agent.chat(msg1)
    print("Reply:", res1["reply"])
    print("Recs:", res1["recommendations"])
    
    print("\n--- Test 2: Recommendation Flow ---")
    msg2 = msg1 + [{"role": "assistant", "content": res1["reply"]},
                   {"role": "user", "content": "It's for a Senior Software Engineer."}]
    res2 = agent.chat(msg2)
    print("Reply:", res2["reply"])
    print("Recs:", res2["recommendations"])
    
    print("\n--- Test 3: Refinement ---")
    msg3 = msg2 + [{"role": "assistant", "content": res2["reply"]},
                   {"role": "user", "content": "Also include personality tests."}]
    res3 = agent.chat(msg3)
    print("Reply:", res3["reply"])
    print("Recs:", len(res3["recommendations"]))
    
    print("\n--- Test 4: Comparison ---")
    msg4 = [{"role": "user", "content": "What is the difference between OPQ and GSA?"}]
    res4 = agent.chat(msg4)
    print("Reply:", res4["reply"])
    
    print("\n--- Test 5: Refusal ---")
    msg5 = [{"role": "user", "content": "Give me legal hiring advice about firing employees."}]
    res5 = agent.chat(msg5)
    print("Reply:", res5["reply"])
