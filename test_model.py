#!/usr/bin/env python3
import json
import sys
from rag import RAGPipeline

def evaluate_response(expected: str, actual: str) -> tuple[bool, float]:
    expected_lower = expected.lower().strip()
    actual_lower = actual.lower().strip()
    
    if expected_lower == "я не знаю":
        found = "я не знаю" in actual_lower
        completeness = 1.0 if found else 0.0
        return found, completeness
    
    found = any(keyword in actual_lower for keyword in expected_lower.split()[:5])
    
    if not found:
        return False, 0.0
    
    expected_words = set(expected_lower.split())
    actual_words = set(actual_lower.split())
    common_words = expected_words & actual_words
    
    completeness = len(common_words) / len(expected_words) if expected_words else 0.0
    completeness = min(completeness, 1.0)
    
    return True, completeness

def main():
    with open("golden_questions.json", "r", encoding="utf-8") as f:
        golden_questions = json.load(f)
    
    rag = RAGPipeline()
    
    total = len(golden_questions)
    correct = 0
    total_completeness = 0.0
    
    print("=" * 60)
    print("Начало тестирования модели")
    print("=" * 60)
    
    for i, (question, expected_answer) in enumerate(golden_questions.items(), 1):
        print(f"\n[{i}/{total}] Вопрос: {question}")
        print(f"    Ожидаемый ответ: {expected_answer[:80]}...")
        
        try:
            actual_answer = rag.answer(question)
        except Exception as e:
            print(f"    ОШИБКА: {e}")
            continue
        
        found, completeness = evaluate_response(expected_answer, actual_answer)
        
        status = "НАЙДЕН" if found else "НЕ НАЙДЕН"
        print(f"    Ответ найден: {status}")
        print(f"    Полнота ответа: {completeness:.2%}")
        print(f"    Фактический ответ: {actual_answer[:100]}...")
        
        if found:
            correct += 1
        total_completeness += completeness
    
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    print(f"Всего вопросов: {total}")
    print(f"Правильных ответов: {correct}/{total} ({correct/total:.1%})")
    print(f"Средняя полнота: {total_completeness/total:.1%}")
    
    return 0 if correct == total else 1

if __name__ == "__main__":
    sys.exit(main())
