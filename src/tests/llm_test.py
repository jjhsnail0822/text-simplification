import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def main():
    model_id = "results/grpo/Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507-trained"

    sampling_params = SamplingParams(max_tokens=1024, temperature=0.0)
    llm = LLM(model_id, max_model_len=2048)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    PROMPT_TEMPLATE = "You are a careful rewrite assistant.\nRewrite the <TEXT> in {lang} so that every word, except proper nouns or proper adjectives, is at or below the {level} vocabulary level.\nReplace or simplify any other words above {level} level with easier alternatives while preserving the original meaning and coherence.\nDo not skip, shorten, or omit any part of the text. Keep sentence count and structure.\nOutput only the fully converted text with no explanations, instructions, or extra words.\n\n<TEXT>\n{shortened_text}"
 
    texts = {
        "English": "Pigeon photography is an aerial photography technique invented in 1907 by the German apothecary Julius Neubronner, who also used pigeons to deliver medications. A homing pigeon was fitted with an aluminium breast harness to which a lightweight time-delayed miniature camera could be attached. Neubronner's German patent application was initially rejected, but was granted in December 1908 after he produced authenticated photographs taken by his pigeons. He publicized the technique at the 1909 Dresden International Photographic Exhibition, and sold some images as postcards at the Frankfurt International Aviation Exhibition and at the 1910 and 1911 Paris Air Shows.\n\nInitially, the military potential of pigeon photography for aerial reconnaissance appeared interesting. Battlefield tests in World War I provided encouraging results, but the ancillary technology of mobile dovecotes for messenger pigeons had the greatest impact. Owing to the rapid development of aviation during the war, military interest in pigeon photography faded and Neubronner abandoned his experiments. The idea was briefly resurrected in the 1930s by a Swiss clockmaker, and reportedly also by the German and French militaries. Although war pigeons were deployed extensively during World War II, it is unclear to what extent, if any, birds were involved in aerial reconnaissance. The United States Central Intelligence Agency (CIA) later developed a battery-powered camera designed for espionage pigeon photography; details of its use remain classified.",
        "Japanese": "シューマンの読書好きは父親譲りで、主として文学と哲学を好んだ。\nシューマンは13歳のとき、当時興味を持った批評や詩、哲学的著作からの引用や自作の劇『精神』（未完）からの断章、両親の文章などを「スクランダー」というペンネームで『美しい黄金色の牧場の葉と花』としてまとめている。\nまた、1825年から1828年の間に書いた自作の文集を「ムルデ河畔のロベルト」というペンネームで『雑録』としてまとめている。このころ、シューマンはゲーテの『ファウスト』をほとんど全部暗記し、友人たちからは「ファウスト」または「メフィスト」などと呼ばれていた。\nこのほか、シューマンが手がけた文学作品として、コリオランを題材にした合唱付きの悲劇『ランデンドルファー兄弟』や喜劇『レオンハルトとマンテリエ』、ジャン・パウルから影響を受けた『6月の晩と7月の昼間』という小説があるが、いずれも未完である。\nシューマンが文学者をめざさず音楽の道を選んだことについて、ブリオンは「シューマンにとって、限界があり、厳密さを欠く文章表現よりも、音楽はずっと豊かで、多様で、陰影があり、緻密な言葉を提供した」と述べている。",
        "Korean": "어린 프리드리히는 1840년대 이래 독일에서 맹위를 떨치던 자유주의 세력 중심의 혼탁한 정국을 겪었다. 당시 자유주의자들은 독일인들의 열광적이고 광범위한 지지를 얻으며 세를 넓혀나가고 있었다. 자유주의자들은 독일의 통일을 희망하였고 입헌군주론자들은 새 헌법을 만들어 모든 인민들의 평등권 보장, 재산 보호, 그리고 기본 인권 보장을 구호로 내세웠다. 즉, 자유주의자들은 인민들의입장을 대변하고 그들의 뜻에 따라 정책을 수립하는 정부를 원하였다. 프리드리히가 17살이 된 1848년, 민족주의자들과 자유주의자들은 독일 전 지역과 서유럽에 걸쳐 대규모의 시위를 주도하였다. 자유주의와 민족주의 세력은 집회와 결사의 자유, 언론의 자유 등의 자유권의 보장과 독일 의회의 수립, 그리고 헌법의 제정을 요구했다. 비록 독일에서의 혁명은 뚜렷한 족적을 남기지는 않았지만, 프리드리히가 어릴 때 목도한 이 자유주의는 훗날 그의 일생에 걸쳐 큰 영향력을 발휘하게 된다.",
        "Chinese": "在18世纪中期，西方文明对其他文明的影响是西方思想家的辩论焦点之一，不少学者认为西方文明优化了其他文明，但也有学者认为西方文明的入侵腐化了其他文明。库克的三次航海探索正值这个辩论的高潮，因此他的航海发现或多或少让西方思想家对地球另一边鲜为西方所知的文化有稍进一步的了解。不过，库克对这个命题并不特别关心，从他的周记所见，也不见得出他对“高贵野蛮人”（Noble Savage）这种在当时盛行的看法有特别的兴趣。在19世纪，波兰裔英国小说家约瑟夫·康拉德曾经对历代航海家和探险家的动机作出比较，他认为库克以前的航海家和探险家主要以“掠夺”（acquisitive）为动机，而库克则主要以“科学”（scientific）为动机，因此两者本质上具有显著的分别。但有其他学者认为，库克三次航海旅程的费用要由英国政府动用公帑承担，这意味出海的计划和目的受到纳税人监察，在这种背景下，库克在旅程途中也不时把新发现的地方宣告为英国领土，因此，如果说他的航海旅程完全不具“掠夺”性质，也不是准确的说法。"
    }

    levels = {
        "English": ["CEFR A1", "CEFR A2", "CEFR B1", "CEFR B2", "CEFR C1", "CEFR C2"],
        "Japanese": ["JLPT N5", "JLPT N4", "JLPT N3", "JLPT N2", "JLPT N1"],
        "Korean": ["TOPIK Level 1", "TOPIK Level 2", "TOPIK Level 3", "TOPIK Level 4", "TOPIK Level 5", "TOPIK Level 6"],
        "Chinese": ["HSK 3.0 Level 1", "HSK 3.0 Level 2", "HSK 3.0 Level 3", "HSK 3.0 Level 4", "HSK 3.0 Level 5", "HSK 3.0 Level 6", "HSK 3.0 Level 7-9"]
    }

    prompts = []
    metadata = []

    for lang, text in texts.items():
        if lang in levels:
            for level in levels[lang]:
                formatted_prompt = PROMPT_TEMPLATE.format(lang=lang, level=level, shortened_text=text)
                messages = [{"role": "user", "content": formatted_prompt}]
                
                full_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
                
                prompts.append(full_prompt)
                metadata.append({
                    "language": lang,
                    "level": level,
                    "original_text": text
                })

    print(f"Generating responses for {len(prompts)} prompts...")

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        entry = metadata[i]
        entry["simplified_text"] = generated_text
        results.append(entry)
        print(f"[{entry['language']} - {entry['level']}] Processed.")

    output_filename = f"results/examples/{model_id.split('/')[-1]}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"All results saved to {output_filename}")

if __name__ == "__main__":
    main()