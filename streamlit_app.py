import streamlit as st
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from typing import Dict, Any, List

# --- CONFIG ---
LT_API_KEY = st.secrets.get("LT_API_KEY")
LT_USERNAME = st.secrets.get("LT_USERNAME")
LT_API_URL = "https://api.languagetoolplus.com/v2/check"

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Essay Assessment AI",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CSS FOR UI ---
st.markdown("""
    <style>
    [data-testid="stSidebar"] {display: none;}
    .main-header {font-size: 2.5rem; color: #4A90E2; text-align: center; margin-bottom: 20px;}
    
    /* Simple Highlight - No Hover */
    .grammar-error {
        background-color: #ffcccc; 
        color: #b30000;
        text-decoration: underline wavy red;
        padding: 0 2px;
        border-radius: 2px;
    }
    
    .ai-score-box {
        background-color: #fff3cd;
        color: #856404;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ffeeba;
        margin-top: 10px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- DATA ---
TOPICS = [
    "Energy",
    "Radiation Tech",
    "Agriculture and Food Security",
    "Water and Environment"
]

SAMPLE_REQUESTS = {
    "Energy": [
        "Discuss the role of nuclear power in the transition to clean energy.",
        "Analyze the challenges of implementing small modular reactors (SMRs).",
    ],
    "Radiation Tech": [
        "Explain how radiation technology is used in modern cancer treatment.",
        "Describe the industrial applications of non-destructive testing (NDT).",
    ],
    "Agriculture and Food Security": [
        "Propose 5 nuclear techniques initiated by IAEA for food safety and success stories.",
        "Evaluate the impact of sterile insect technique (SIT) on pest control.",
    ],
    "Water and Environment": [
        "Assess the use of isotope hydrology in managing groundwater resources.",
        "Discuss methods for monitoring marine pollution using nuclear techniques."
    ]
}

def get_sample_essay_text_dummy(topic, request):
    return f"""This is a generated sample essay regarding {topic}.
    
The request was: {request}

Nuclear techniques have played a pivotal role in {topic.lower()}. First, the IAEA has initiated several programs. One example is the use of irradiation to preserve food. This method is highly accurate in killing bacteria without making the food radioactive.

However, some people argues that it is dangerous. This claims is inaccurate. The process is strictly controlled.
    
Furthermore, in terms of development, the technology has shown profound results in increasing shelf life. The logic follows a scientific path from research to implementation.

(This is a short placeholder sample to demonstrate the functionality)."""

def get_sample_essay_text(topic, request):
    return f"""The International Atomic Energy Agency (IAEA), in collaboration with the Food and Agriculture Organization (FAO), has initiated and promoted several nuclear techniques to enhance food safety and protect public health worldwide. These techniques help detect contamination, prevent foodborne diseases, and support safe international food trade.

One of the most significant contributions is food irradiation. This technique uses controlled doses of ionizing radiation to eliminate harmful microorganisms such as bacteria and parasites, delay spoilage, and extend shelf life without compromising nutritional quality. A major success story is Vietnam, where IAEA-supported irradiation facilities have enabled safer fruit exports by meeting international phytosanitary standards, reducing post-harvest losses and increasing market access.

Another important technique is stable isotope analysis, which is used to verify food authenticity and trace its geographical origin. This method helps detect food fraud and adulteration. For example, in Bangladesh, isotope techniques revealed widespread honey adulteration, prompting corrective actions and improving consumer protection.

The IAEA has also promoted nuclear and related analytical techniques for detecting food contaminants, including pesticide residues, veterinary drug residues, mycotoxins, and heavy metals. Through training and laboratory support, countries such as Uganda have strengthened their food safety laboratories, enabling more reliable monitoring of food hazards in line with international standards.

A further application is the use of ionizing radiation for phytosanitary pest control. This technique ensures that exported agricultural products are free from invasive pests without relying on chemical fumigants. It has helped many countries comply with quarantine regulations, facilitating safer global food trade.

Finally, the IAEA supports advanced nuclear-based laboratory methods, such as isotope ratio mass spectrometry, for rapid and accurate food safety analysis. These techniques improve the efficiency of food inspections and help national authorities enforce food safety regulations.

In conclusion, IAEA-initiated nuclear techniques have played a vital role in improving food safety, reducing contamination, preventing fraud, and strengthening international food trade. Their successful application across many countries demonstrates how nuclear science contributes directly to public health and food security."""

# --- CLASS: GRAMMAR CHECKER (LIGHTWEIGHT) ---
class SimpleLanguageToolChecker:
    def __init__(self, api_key: str, username: str, api_url: str):
        self.api_key = api_key
        self.username = username
        self.api_url = api_url
    
    def check_text(self, text: str) -> List[Dict[str, Any]]:
        """Call API and return list of errors."""
        try:
            data = {
                "text": text,
                "language": "en-US",
                "username": self.username,
                "apiKey": self.api_key,
                "level": "picky"
            }
            response = requests.post(self.api_url, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()
            
            errors = []
            for match in result.get("matches", []):
                replacements = [r["value"] for r in match.get("replacements", [])][:3]
                errors.append({
                    "offset": match["offset"],
                    "length": match["length"],
                    "replacements": replacements,
                    "bad_word": text[match["offset"]:match["offset"]+match["length"]]
                })
            return errors
        except Exception:
            return []

def render_simple_highlight(text: str, errors: List[Dict[str, Any]]) -> str:
    """Chỉ bôi màu từ sai, không chèn tooltip."""
    if not errors:
        return text
    
    # Sort reverse to handle offsets correctly
    errors.sort(key=lambda x: x["offset"], reverse=True)
    
    highlighted_text = text
    for err in errors:
        start = err["offset"]
        end = start + err["length"]
        bad_segment = highlighted_text[start:end]
        
        # Simple HTML span class
        replacement = f'<span class="grammar-error">{bad_segment}</span>'
        highlighted_text = highlighted_text[:start] + replacement + highlighted_text[end:]
        
    return highlighted_text.replace("\n", "<br>")

# --- LLM SETUP & PROMPT ---
def get_llm():
    try:
        access_key = st.secrets.get("AWS_ACCESS_KEY_ID") 
        secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY") 
        
        if not access_key or not secret_key:
            st.error("AWS Credentials not found. Please check Streamlit Secrets.")
            return None

        llm = ChatBedrock(
            aws_access_key_id=access_key,  # Đã thêm dấu phẩy
            aws_secret_access_key=secret_key,
            region_name="us-east-1",       # THÊM DÒNG NÀY (Bắt buộc)
            provider="anthropic",
            model_id="arn:aws:bedrock:us-east-1:605134429290:inference-profile/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
            temperature=0
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Bedrock: {e}")
        return None

def build_assessment_prompt(topic, request, essay, check_ai, vietsub=False):
    # Logic: Xây dựng chuỗi format output dựa trên các lựa chọn
    if vietsub:
        system_role = "Bạn là chuyên gia đánh giá học thuật của IAEA. Hãy trả lời BẰNG TIẾNG VIỆT."
        
        # Xây dựng danh sách các mục output bắt buộc
        output_structure = """
            - Phản hồi yêu cầu: (1~2 câu đánh giá)
            - Độ chính xác thông tin: (2~3 câu xác thực. In đậm nếu sử dụng các từ như **chính xác**/**không chính xác**)
            - Phát triển ý tưởng: (1~2 câu. In đậm nếu sử dụng các từ như **Sâu sắc**/**Hời hợt**)
            - Sự mạch lạc: (1~2 câu)
            - Kết luận: (2~3 câu)
            - Đánh giá tổng quan: (Chỉ trả về 1 trong các giá trị: Kém / Trung bình / Khá / Tốt / Xuất sắc)"""
        
        if check_ai:
            output_structure += "\n            - Phát hiện AI: (Trả về ước tính theo % từ 1 đến 100)"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_role),
            ("human", f"""
            **Chủ đề:** {topic}
            **Yêu cầu:** {request}
            **Bài làm:** {essay}
            
            ---
            **HƯỚNG DẪN:**
            Đánh giá bài luận và trả về kết quả theo ĐÚNG CẤU TRÚC sau (Không thay đổi tên mục):
            {output_structure}
            """)
        ])
    else:
        system_role = "You are a strict academic evaluator for the IAEA."
        
        output_structure = """
            - Task Response: (1~2 sentences)
            - Information Accuracy: (2~3 sentences. Bold **accurate**, **inaccurate**)
            - Idea Development: (1~2 sentences. Bold **Profound** or **Superficial**)
            - Coherence: (1~2 sentences)
            - Summary: (2~3 sentences)
            - Final Evaluation: (Choose one: Poor / Average / Good / Excellent / Outstanding)"""
        
        if check_ai:
             output_structure += "\n            - AI Plagiarism: (Estimated %, from 0 to 100)"

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_role),
            ("human", f"""
            **Topic:** {topic}
            **Request:** {request}
            **Essay:** {essay}
            
            ---
            **INSTRUCTIONS:**
            Evaluate and provide output in the EXACT format below:
            {output_structure}
            """)
        ])
    return prompt_template

# --- MAIN ---
def main():
    st.markdown('<div class="main-header">📝 Essay Assessment Suport ver 0.1</div>', unsafe_allow_html=True)

    # Inputss
    selected_topic = st.selectbox("Select Topic", TOPICS)
    req_options = SAMPLE_REQUESTS.get(selected_topic, []) + ["Custom Request..."]
    selected_req_option = st.selectbox("Select or Enter Request", req_options)
    
    final_request = st.text_input("Enter Request:", value="") if selected_req_option == "Custom Request..." else selected_req_option

    # Essay State
    if "essay_text_area" not in st.session_state:
        st.session_state["essay_text_area"] = ""

    if st.button("Use Sample Essay"):
        st.session_state["essay_text_area"] = get_sample_essay_text(selected_topic, final_request)
        st.rerun()

    essay_input = st.text_area("Student Submission:", height=300, key="essay_text_area")

    # Options
    with st.expander("Advanced Analysis Options"):
        c1, c2, c3 = st.columns(3)
        check_grammar = c1.checkbox("Check Grammar", False)
        check_ai = c2.checkbox("Check AI Plagiarism", False)
        vietsub_mode = c3.checkbox("Vietsub (Tiếng Việt)", False)

    # Analyze
    if st.button("🔍 Analyze Essay", type="primary", use_container_width=True):
        if not final_request or not essay_input.strip():
            st.warning("Please provide request and essay.")
            return

        # 1. LLM Analysis
        llm = get_llm()
        llm_response_text = ""
        if llm:
            with st.spinner('Analyzing...'):
                prompt = build_assessment_prompt(selected_topic, final_request, essay_input, check_ai, vietsub_mode)
                chain = prompt | llm | StrOutputParser()
                try:
                    llm_response_text = chain.invoke({})
                    print(llm_response_text)
                except Exception as e:
                    st.error(f"LLM Error: {e}")

        # 2. Grammar Check (Lightweight)
        grammar_errors = []
        if check_grammar:
            with st.spinner('Checking grammar...'):
                checker = SimpleLanguageToolChecker(LT_API_KEY, LT_USERNAME, LT_API_URL)
                grammar_errors = checker.check_text(essay_input)

        # 3. Display Results
        st.divider()
        st.subheader("Evaluation Results")
        
        # --- PARSING LOGIC ---
        # Map keywords for both languages
        section_map = {
            "Task Response": ["Task Response", "Phản hồi yêu cầu"],
            "Information Accuracy": ["Information Accuracy", "Độ chính xác thông tin"],
            "Idea Development": ["Idea Development", "Phát triển ý tưởng"],
            "Coherence": ["Coherence", "Sự mạch lạc"],
            "Summary": ["Summary", "Kết luận"],
            "Final Evaluation": ["Final Evaluation", "Đánh giá tổng quan"],
            "AI Plagiarism": ["AI Plagiarism", "Phát hiện AI"]
        }
        
        parsed_data = {k: "" for k in section_map}
        current_section = None
        
        lines = llm_response_text.split('\n')
        print(lines)
        
        for line in lines:
            clean_line = line.strip()
            if not clean_line: continue
            
            is_new_section = False
            for section_key, keywords in section_map.items():
                norm_line = clean_line.replace('*', '').replace('-', '').replace('#', '').strip().lower()
                for kw in keywords:
                    if norm_line.startswith(kw.lower()):
                        current_section = section_key
                        is_new_section = True
                        if ':' in clean_line:
                            content_part = clean_line.split(':', 1)[1].strip()
                            if content_part:
                                parsed_data[current_section] += content_part + " "
                        break
                if is_new_section: break
            
            if not is_new_section and current_section:
                parsed_data[current_section] += clean_line + " "

        # --- RENDER LLM OUTPUT ---
        def show_section(title, content_key):
            content = parsed_data.get(content_key, "").strip()
            # Clean leading markers
            content = content.lstrip('*').lstrip('-').strip()
            if content:
                st.markdown(f"**{title}:** {content}")

        if vietsub_mode:
            show_section("Phản hồi yêu cầu", "Task Response")
            show_section("Độ chính xác thông tin", "Information Accuracy")
            show_section("Phát triển ý tưởng", "Idea Development")
            show_section("Sự mạch lạc", "Coherence")
            show_section("Kết luận", "Summary")
        else:
            show_section("Task Response", "Task Response")
            show_section("Information Accuracy", "Information Accuracy")
            show_section("Idea Development", "Idea Development")
            show_section("Coherence", "Coherence")
            show_section("Summary", "Summary")

        # Final Eval & AI
        final_eval = parsed_data.get("Final Evaluation", "").strip()
        ai_score = parsed_data.get("AI Plagiarism", "").strip()
        
        # Hiển thị Final Eval
        if final_eval:
            # Clean up if duplication occurs (LLM quirk)
            final_eval = final_eval.lstrip('*').lstrip(":").strip()
            st.markdown(f"### Final Evaluation: :blue[{final_eval}]")
            
        # Hiển thị AI Box (Nếu có dữ liệu AI)
        if check_ai and ai_score:
            ai_score = ai_score.lstrip('*').lstrip(":").strip()
            st.markdown(f'<div class="ai-score-box">⚠️ AI Detection: {ai_score}</div>', unsafe_allow_html=True)

        # 4. Show Grammar (Simple List)
        if check_grammar:
            st.divider()
            st.subheader("Grammar Check")
            if grammar_errors:
                st.write(f"Found {len(grammar_errors)} issues:")
                
                # Render Highlighted Text
                html_view = render_simple_highlight(essay_input, grammar_errors)
                st.markdown(
                    f'<div style="background-color: #f0f2f6; color: black; padding: 15px; border-radius: 5px; line-height: 1.6;">{html_view}</div>', 
                    unsafe_allow_html=True
                )
                
                # Render List of Errors below
                error_data = []
                for e in grammar_errors:
                    sugg = ", ".join(e['replacements']) if e['replacements'] else "None"
                    error_data.append({"Error": e['bad_word'], "Suggestions": sugg})
                
                st.table(error_data) # Simple, clean table
            else:
                st.success("No grammar errors found.")

if __name__ == "__main__":
    main()
