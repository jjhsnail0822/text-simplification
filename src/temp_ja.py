import pandas as pd
import re

def find_non_japanese_chars_in_csv(file_path):
    """
    CSV 파일의 'Kanji'와 'Hiragana' 열을 분석하여,
    일반적인 일본어 문자 외의 기호가 포함된 단어를 찾고 해당 기호를 출력합니다.

    Args:
        file_path (str): 분석할 CSV 파일의 경로
    """
    try:
        # 1. CSV 파일 읽기
        df = pd.read_csv(file_path, encoding='utf-8')
        # 열 이름의 앞뒤 공백 제거
        df.columns = df.columns.str.strip()
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return
    except Exception as e:
        print(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return

    # 2. 검사할 열 목록 (대소문자 수정)
    columns_to_check = ['Kanji', 'Hiragana']

    # 3. 허용하지 않을 문자 패턴 정의 (정규표현식)
    # 한자, 히라가나, 가타카나, ・, ー, 々, 공백을 제외한 모든 문자
    non_japanese_pattern = re.compile(r'[^一-龯ぁ-んァ-ン・ー々\s]')

    print(f"'{file_path}' 파일 분석 시작...")
    print("-" * 30)

    found_issues = False

    # 4. 데이터프레임의 각 행을 순회하며 검사
    for index, row in df.iterrows():
        for col_name in columns_to_check:
            # 해당 열이 데이터프레임에 없으면 건너뛰기
            if col_name not in df.columns:
                continue

            # 셀 값이 문자열이 아니거나 비어있으면 건너뛰기
            word = row[col_name]
            if not isinstance(word, str) or not word:
                continue

            # 허용되지 않는 기호들을 한 번에 찾기
            found_symbols = non_japanese_pattern.findall(word)
            
            # 비정상 기호가 발견된 경우 결과 출력
            if found_symbols:
                found_issues = True
                # 중복된 기호는 한 번만 보여주기
                unique_symbols = sorted(list(set(found_symbols)))
                print(f"행 {index+2}: 비정상 기호 발견!")
                print(f"  - 열: '{col_name}'")
                print(f"  - 전체 단어: '{word}'")
                print(f"  - 포함된 기호: {', '.join(unique_symbols)}")
                print()

    if not found_issues:
        print("분석 완료: 'Kanji' 및 'Hiragana' 열에서 일본어 외의 기호가 발견되지 않았습니다.")
    else:
        print("-" * 30)
        print("분석이 완료되었습니다.")


# --- 실행 부분 ---
# 여기에 실제 CSV 파일 경로를 입력하세요.
csv_file_path = 'data/wordlist_ja_raw.csv' 

# 함수 호출
find_non_japanese_chars_in_csv(csv_file_path)