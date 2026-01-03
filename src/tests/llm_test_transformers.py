import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # 모델 경로
    model_id = "results/grpo/Qwen3-4B-Instruct-2507-2.0-1.0-1.0-new/checkpoint-600-merged"

    # 프롬프트 및 데이터 설정
    PROMPT_TEMPLATE = "You are a careful rewrite assistant.\nRewrite the <TEXT> in {lang} so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\nReplace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\nDo not skip, shorten, or omit any part of the text. Keep sentence count and structure.\nOutput only the fully converted text with no explanations, instructions, or extra words.\n\n<TEXT>\n{shortened_text}"
    lang = "Korean"
    # shortened_text='Philosophy (from old Greek words meaning "love of wisdom") is the careful study of very basic questions about life, reason, knowledge, value, mind, and language. It is a way of thinking that looks closely at its own methods and ideas.\nIn the past, many sciences, like physics and psychology, were part of philosophy. Today, they are seen as different areas of study. Important traditions in the history of philosophy include Western, Arabic–Persian, Indian, and Chinese ways of thought. Western philosophy started in Ancient Greece and has many smaller areas. A key topic in Arabic–Persian philosophy is the link between reason and faith. Indian philosophy joins the problem of how to reach spiritual freedom with the study of reality and knowledge. Chinese philosophy often looks at practical matters like right social behavior, government, and personal growth.\nMain areas of philosophy are knowledge study, ethics, logic, and study of reality. The study of knowledge asks what knowledge is and how we can get it. Ethics studies rules of good and bad and what right action is. Logic is the study of correct thinking and shows how strong arguments are different from weak ones. The study of reality looks at the most general parts of the world, being, objects, and their qualities. Other parts are beauty, language, mind, religion, science, math, history, and politics. In each area, there are different groups that support other ideas, rules, or ways.\nPhilosophers use many ways to reach knowledge. These include looking at ideas, using common sense, testing thoughts in the mind, looking at everyday language, telling about experience, and asking deep questions. Philosophy is linked to many other fields, like science, math, business, law, and news writing. It gives a wide view and studies the main ideas of these fields. It also looks at their methods and moral effects.'
    shortened_text = "철학(Philosophy)이라는 용어는 고대 그리스어의 필로소피아(φιλοσοφία, 지혜에 대한 사랑)에서 유래하였는데, 여기서 지혜는 일상생활에서의 실용하는 지식이 아닌 인간 자신과 그것을 둘러싼 세계를 관조하는 지식을 뜻한다. 이를테면 세계관, 인생관, 가치관이 포함된다. 이런 일반적인 의미로서의 철학은 어느 문화권에나 오래전부터 존재하여 왔다. 고대 그리스에서는 사실 학문 그 자체를 논하는 단어였고 전통상으로는 세계와 인간과 사물과 현상의 가치와 궁극적인 뜻을 향한 본질적이고 총체적인 천착을 뜻했다. 동양의 서구화 이후 철학은 대체로 고대 그리스 철학에서 시작하는 서양철학 일반을 지칭하기도 하나 철학 자체는 동서로 분리되지 않는다. 이에 더하여 현대 철학은 철학에 기초한 사고인 전제나 문제 명확화, 개념 엄밀화, 명제 간 관계 명료화를 이용해 제 주제를 논하는 언어철학이나 논리학등에 상당한 비중을 두고 있다."
    # shortened_text = "歴史上、物理学や心理学など、多くの個別的な科学が哲学の一部から発生した。しかし、現代的には、これらの科学は哲学とは異なる分野として扱われている。歴史上、影響力のある哲学の伝統としては、西洋哲学、アラブ・ペルシア哲学、インド哲学、中国哲学などがある。西洋哲学は古代ギリシアに起源を持ち、哲学における幅広い下位分野をカバーする。アラブ・ペルシア哲学における主要なトピックは理性と啓示の関係であり、インド哲学はどのようにして悟りに達するかの精神的な問題と、現実の本質や知識にたどり着く方法の探求を結びつける。中国哲学は主に正しい社会的行動や統治、そして自己修養（英語版）に関する実践的な問題に焦点を置く。"
    # shortened_text = "哲学是研究普遍的、基本问题的学科，包括存在、知识、价值、理智、心灵、语言、人生、道德等领域。哲学与其他学科不同之处在於哲学有独特之思考方式，例如批判的方式、通常是系统化的方法，并以理性论证为基础。从历史上看，许多单独的学科，例如物理学、生物学等自然科学，或法学、政治学、心理学等社会科学，都曾被视作哲学的一部分或其分支学科。直至其得到后续发展后，才逐渐被视作现代意义上的独立学科。"
    LEVEL = "TOPIK Level 1"
    
    formatted_prompt = PROMPT_TEMPLATE.format(lang=lang, level=LEVEL, shortened_text=shortened_text)
    messages = [{"role": "user", "content": formatted_prompt}]

    print(f"Loading model from {model_id}...")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 모델 로드 (vllm 대신 transformers 사용)
    # device_map="auto"를 사용하여 GPU가 있으면 자동으로 할당합니다.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )

    # 채팅 템플릿 적용
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 입력 데이터 토큰화
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print("Generating...")
    
    # 텍스트 생성
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.0, # Greedy decoding을 위해 0.0 설정 (do_sample=False와 함께 사용)
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    # 생성된 토큰만 디코딩 (입력 프롬프트 부분 제외)
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("Generated Text:")
    print(response)

if __name__ == "__main__":
    main()
