import math
import re
from typing import List, Dict, Tuple, Any, Optional, Type
from functools import lru_cache
import asyncio
import instructor
from openai import OpenAI
import time
import ollama
import numpy as np
from scipy.spatial.distance import cosine
from pydantic import BaseModel, Field, ValidationError

LLM_MODEL = 'gemma2'
EMBEDDING_MODEL = 'mxbai-embed-large'

class DetailedCritique(BaseModel):
    flaws: List[str] = Field(..., description="List of identified flaws and areas for improvement")
    score: float = Field(..., description="Numerical score between -100 and +100")
    justification: str = Field(..., description="Explanation of the reasoning behind the score")

class StructuredFeedback(BaseModel):
    reasoning_improvements: List[str] = Field(..., description="Specific ways to improve the reasoning process")
    factual_corrections: List[str] = Field(..., description="Factual errors that need to be addressed")
    clarity_enhancements: List[str] = Field(..., description="Suggestions to make the answer clearer and more concise")
    content_expansion: List[str] = Field(..., description="Areas where the answer could be more comprehensive")
    key_points: List[str] = Field(..., description="Important points or perspectives missing from the current answer")

class RefinedAnswer(BaseModel):
    reasoning_process: str = Field(..., description="Explanation of the updated thought process")
    verification: str = Field(..., description="Confirmation that all feedback points have been addressed")
    final_answer: str = Field(..., description="The refined and improved answer")

class WeakHints(BaseModel):
    major_flaws: List[str] = Field(..., description="Major flaws in reasoning")
    factual_inaccuracies: List[str] = Field(..., description="Factual inaccuracies in the answer")
    structure_clarity: str = Field(..., description="Assessment of the structure and clarity of the answer")
    depth_breadth: str = Field(..., description="Evaluation of the depth and breadth of the content")
    missing_points: List[str] = Field(..., description="Missing key points or perspectives")

class BetterAnswer(BaseModel):
    reasoning_process: str = Field(..., description="Explanation of the thought process")
    feedback_addressing: str = Field(..., description="Description of how the feedback is being addressed")
    verification: str = Field(..., description="Double-check of the refined answer")
    final_answer: str = Field(..., description="The refined answer")

class LLM:
    def __init__(self, model=LLM_MODEL):
        self.model = model
        self.client = instructor.from_openai(
            OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # required, but unused
            ),
            mode=instructor.Mode.JSON,
        )

async def generate(prompt: str, response_model: Optional[Type[BaseModel]] = None, history: List[str] = [], timeout: int = 150) -> Tuple[Any, List[str]]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("プロンプトは空でない文字列である必要があります。")
    
    if not isinstance(history, list):
        raise ValueError("履歴はリストである必要があります。")
    
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("タイムアウトは正の整数である必要があります。")

    llm = LLM()
    
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for i, message in enumerate(history):
        role = "user" if i % 2 == 0 else "assistant"
        messages.append({"role": role, "content": message})
    messages.append({"role": "user", "content": prompt})

    try:
        start_time = time.time()
        if response_model:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm.client.chat.completions.create,
                    model=llm.model,
                    messages=messages,
                    response_model=response_model
                ),
                timeout=timeout
            )
        else:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    llm.client.chat.completions.create,
                    model=llm.model,
                    messages=messages,
                    response_model=str
                ),
                timeout=timeout
            )
        
        updated_history = history + [prompt, str(response)]
        
        return response, updated_history

    except asyncio.TimeoutError:
        raise TimeoutError(f"応答の生成が{timeout}秒以内に完了しませんでした。")
    except Exception as e:
        if "connection" in str(e).lower():
            raise ConnectionError(f"LLMサービスとの接続に問題が発生しました: {str(e)}")
        else:
            raise

@lru_cache(1024)
def extract_label(text: str, type: str = '') -> Optional[str]:
    text = text.strip().lower()

    if not text:
        return None

    if type == 'numeric':
        match = re.search(r'-?\d+(\.\d+)?', text)
        return match.group() if match else None

    elif type == 'choice':
        match = re.search(r'\b[a-d]\b', text)
        return match.group().upper() if match else None

    elif type == 'yesno':
        if 'yes' in text:
            return 'Yes'
        elif 'no' in text:
            return 'No'
        return None

    elif type == 'formula':
        match = re.search(r'\$(.+?)\$', text)
        return match.group(1) if match else None

    else:  # type == 'text' or その他
        match = re.search(r'the answer is[:\s]*(.+)', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        sentences = text.split('.')
        return sentences[-1].strip() if sentences else None

def check(gt: str, ans: str, answer_type: str = 'text', tolerance: float = 1e-6) -> bool:
    gt = gt.strip()
    ans = ans.strip()

    if answer_type == 'numeric':
        return check_numeric(gt, ans, tolerance)
    elif answer_type == 'choice':
        return check_choice(gt, ans)
    elif answer_type == 'yesno':
        return check_yesno(gt, ans)
    elif answer_type == 'formula':
        return check_formula(gt, ans)
    else:  # 'text' or その他
        return check_text(gt, ans)

def check_numeric(gt: str, ans: str, tolerance: float) -> bool:
    try:
        gt_value = float(gt)
        ans_value = float(ans)
        return math.isclose(gt_value, ans_value, rel_tol=tolerance)
    except ValueError:
        return False

def check_choice(gt: str, ans: str) -> bool:
    return gt.lower() == ans.lower()

def check_yesno(gt: str, ans: str) -> bool:
    return gt.lower() == ans.lower()

def check_formula(gt: str, ans: str) -> bool:
    return gt.replace(' ', '') == ans.replace(' ', '')

def get_embedding(text: str) -> List[float]:
    """
    テキストのembeddingを取得する関数
    """
    embedding = ollama.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return embedding['embedding']

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    2つのベクトル間のコサイン類似度を計算する関数
    """
    return 1 - cosine(v1, v2)

def check_text(gt: str, ans: str, similarity_threshold: float = 0.8) -> bool:
    """
    テキストの意味的な類似性を考慮した高度な比較を行う関数
    
    Args:
    gt (str): 正解（Ground Truth）
    ans (str): 生成された回答
    similarity_threshold (float): 類似度の閾値（この値以上で正解とみなす）

    Returns:
    bool: 回答が正解と十分に類似していればTrue、そうでなければFalse
    """
    # 前処理：小文字化と空白の正規化
    gt = ' '.join(gt.lower().split())
    ans = ' '.join(ans.lower().split())

    # 完全一致の場合はTrueを返す
    if gt == ans:
        return True

    # embeddingの取得
    gt_embedding = get_embedding(gt)
    ans_embedding = get_embedding(ans)

    # コサイン類似度の計算
    similarity = cosine_similarity(gt_embedding, ans_embedding)

    # 類似度が閾値以上ならTrueを返す
    return similarity >= similarity_threshold

# MCTSrの主要クラス
class MCTSr:
    def __init__(self, query: str, ground_truth: str, max_iter: int = 16, ans_format: str = ''):
        self.query = query
        self.ground_truth = ground_truth
        self.max_iter = max_iter
        self.ans_format = ans_format
        
        self.to_explore: List[str] = []
        self.to_explore_reward: Dict[str, List[float]] = {}
        self.history_bank: Dict[str, Tuple[str, ...]] = {}
        self.hints_bank: Dict[str, List[str]] = {}
        self.ucb_bank: Dict[str, float] = {}
        self.fathers: Dict[str, str] = {}
        self.childs: Dict[str, List[str]] = {}
        self.hints_reward_imp_bank: Dict[str, List[Tuple[str, float, str]]] = {}

        # 改善点5: 終了条件の拡張
        self.early_stop_threshold = 0.01  # 改善率の閾値
        self.max_depth = 10  # 最大探索深度
        self.logits_threshold = 0.95  # 言語モデルのロジットに基づく閾値

    async def sampling_reward(self, answer: str) -> None:
        if answer not in self.to_explore_reward:
            self.to_explore_reward[answer] = []
        reward = await self.cal_reward(self.query, answer)
        self.to_explore_reward[answer].append(reward)

    async def cal_reward(self, question: str, ans: str) -> float:
        prompt = f"""
        Question: {question}
        Answer: {ans}
        
        Task:
        1. Analyze this answer rigorously and critically.
        2. Identify and explain every flaw, no matter how small.
        3. Be extremely strict in your evaluation - do not hesitate to point out imperfections.
        4. Assign a score between -100 and +100, where:
            - -100 represents a completely incorrect or irrelevant answer
            + 100 represents a theoretically perfect answer (note: this score should rarely, if ever, be given)
        """
        response, _ = await generate(prompt, response_model=DetailedCritique)
        
        score = response.score
        print(f'{score=}')
        if score > 95:
            score = 95 + (score - 95) * 0.1
        return score

    def add_to_hints_bank(self, hints: WeakHints, weak_answer: str) -> None:
        if weak_answer not in self.hints_bank:
            self.hints_bank[weak_answer] = []
        self.hints_bank[weak_answer].append(hints.model_dump_json())

    def add_to_childs(self, father: str, child: str) -> None:
        if father not in self.childs:
            self.childs[father] = []
        self.childs[father].append(child)

    def add_to_hints_reward_imp_bank(self, hints: WeakHints, weak_answer: str, reward: float, answer: str) -> None:
        if weak_answer not in self.hints_reward_imp_bank:
            self.hints_reward_imp_bank[weak_answer] = []
        self.hints_reward_imp_bank[weak_answer].append((hints.model_dump_json(), reward, answer))

    def filter_mature_node(self, max_expand: int = 3) -> List[str]:
        # 改善点7: 候補ノードの選択の改善
        filterd_to_explore = []
        avg_reward = self.calculate_avg_reward()

        for node in self.to_explore:
            child_count = len(self.childs.get(node, []))
            max_child_reward = max([avg_reward.get(child, float('-inf')) for child in self.childs.get(node, [])] + [float('-inf')])
            node_reward = avg_reward.get(node, float('-inf'))

            if child_count < max_expand or max_child_reward <= node_reward:
                filterd_to_explore.append(node)
        
        return filterd_to_explore

    def calculate_avg_reward(self) -> Dict[str, float]:
        return {node: (min(rewards) + sum(rewards) / len(rewards)) / 2 for node, rewards in self.to_explore_reward.items()}

    def get_best_explore_from_ucb(self, to_explore: List[str]) -> str:
        best_node = max(to_explore, key=lambda node: self.ucb_bank.get(node, float('-inf')))
        return best_node

    def compute_ucb(self, r_c: float, N_n: int, N_c: int, C: float = 1.4, epsilon: float = 1e-5) -> float:
        return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + epsilon))

    def update_ucb(self, C: float = 1.4) -> None:
        # 改善点8: 探索と活用のバランスの調整
        visit_count = {node: len(rewards) for node, rewards in self.to_explore_reward.items()}
        avg_reward = self.calculate_avg_reward()

        leaves = set(self.to_explore) - set(self.fathers.values())
        
        for leaf in leaves:
            self.ucb_bank[leaf] = self.compute_ucb(
                avg_reward[leaf],
                len(self.to_explore_reward.get(self.fathers.get(leaf, None), [])),
                len(self.to_explore_reward.get(leaf, [])),
                C=C * (1 + math.log(len(self.to_explore) + 1) / 10)  # Cを動的に調整
            )
        
        nodes_to_update = list(leaves)
        while nodes_to_update:
            new_nodes_to_update = set()
            for node in nodes_to_update:
                father = self.fathers.get(node)
                if father is not None:
                    if father not in self.ucb_bank:
                        new_nodes_to_update.add(father)
                    if father in avg_reward:
                        child_reward = max([avg_reward[child] for child in self.childs[father]])
                        father_reward = (avg_reward[father] + child_reward) / 2
                        self.ucb_bank[father] = self.compute_ucb(
                            father_reward,
                            len(self.to_explore_reward.get(self.fathers.get(father, None), [])),
                            len(self.to_explore_reward.get(father, [])),
                            C=C * (1 + math.log(len(self.to_explore) + 1) / 10)  # Cを動的に調整
                        )
            nodes_to_update = list(new_nodes_to_update)

    async def step(self, weak_answer: str) -> Tuple[WeakHints, BetterAnswer, List[str]]:
        # 改善点6: 自己改善プロセスの強化
        history_list = list(self.history_bank[weak_answer])
        hints, history = await self.get_weak_hints(self.query, weak_answer, history=history_list)
        better_answer, history = await self.get_better_answer(self.query, weak_answer, hints, history=history)
        
        # 構造化されたフィードバックの活用
        structured_feedback = await self.generate_structured_feedback(hints, better_answer.final_answer)
        improved_answer, history = await self.refine_answer_with_feedback(self.query, better_answer.final_answer, structured_feedback, history=history)
        
        return hints, improved_answer, history

    async def get_weak_hints(self, question: str, weak_answer: str, history: List[str] = []) -> Tuple[WeakHints, List[str]]:
        query = f"""
        Question: {question}
        Weak Answer: {weak_answer}

        Task:
        1. Analyze the weak answer critically and thoroughly.
        2. Provide detailed feedback, highlighting every flaw and imperfection.
        3. Consider all possible aspects that could be improved.
        4. Be comprehensive in your critique.

        Your response should follow the structure defined in the WeakHints model.
        """
        response, updated_history = await generate(query, response_model=WeakHints, history=history)
        return response, updated_history

    async def get_better_answer(self, question: str, weak_answer: str, hint: WeakHints, history: List[str] = []) -> Tuple[BetterAnswer, List[str]]:
        query = f"""
        Question: {question}
        Previous Answer: {weak_answer}
        Feedback: {hint.model_dump_json()}

        Task:
        Refine the previous answer according to the provided feedback. Your response should follow the structure defined in the BetterAnswer model, which includes:
        1. reasoning_process: Explanation of your updated thought process
        2. feedback_addressing: Specific description of how you're addressing each point of feedback
        3. verification: A double-check of your refined answer
        4. final_answer: The refined and improved answer

        Remember to think through each step carefully as you refine your response.
        """
        try:
            response, updated_history = await generate(query, response_model=BetterAnswer, history=history)
        except ValidationError as e:
            print(f"Validation error: {e}")
            # Implement fallback logic or retry mechanism here
            # For example, you could try to extract partial information or request a new response
        return response, updated_history

    async def generate_structured_feedback(self, hints: WeakHints, answer: str) -> StructuredFeedback:
        query = f"""
        Original Hints: {hints.model_dump_json()}
        Current Answer: {answer}

        Task:
        Based on the original hints and the current answer, generate structured feedback.
        """
        feedback, _ = await generate(query, response_model=StructuredFeedback)
        return feedback

    async def refine_answer_with_feedback(self, question: str, answer: str, structured_feedback: StructuredFeedback, history: List[str] = []) -> Tuple[RefinedAnswer, List[str]]:
        query = f"""
        Question: {question}
        Current Answer: {answer}
        Structured Feedback:
        {structured_feedback.model_dump_json(indent=2)}

        Task:
        Refine the current answer based on the structured feedback provided.
        """
        return await generate(query, response_model=RefinedAnswer, history=history)

    async def main_loop(self) -> Tuple[List[WeakHints], List[RefinedAnswer], List[str], Dict[str, List[float]], Dict[str, List[str]], Dict[str, Tuple[str, ...]], Dict[str, List[Tuple[str, float, str]]], Dict[str, str], Dict[str, List[str]], Dict[str, float]]:
        weak_answer, history = await self.get_weak_answer()
        self.history_bank[weak_answer] = tuple(history)
        self.to_explore = [weak_answer]
        self.childs[weak_answer] = []
        self.fathers[weak_answer] = None
        await self.sampling_reward(weak_answer)

        hints_list = []
        answers_list = [RefinedAnswer(reasoning_process="Initial answer", verification="No verification", final_answer=weak_answer)]

        self.update_ucb()

        for iteration in range(self.max_iter):
            filtered_to_explore = self.filter_mature_node()
            weak_answer = self.get_best_explore_from_ucb(filtered_to_explore)
            await self.sampling_reward(weak_answer)

            hints, refined_answer, history = await self.step(weak_answer)
            self.add_to_hints_bank(hints, weak_answer)
            self.history_bank[refined_answer.final_answer] = tuple(history)
            self.to_explore.append(refined_answer.final_answer)
            await self.sampling_reward(refined_answer.final_answer)
            self.fathers[refined_answer.final_answer] = weak_answer
            self.childs[refined_answer.final_answer] = []
            self.add_to_childs(weak_answer, refined_answer.final_answer)
            answers_list.append(refined_answer)
            hints_list.append(hints)

            # 改善点5: 終了条件の拡張
            if self.check_termination(refined_answer.final_answer, iteration):
                break

            self.update_ucb()
            self.add_to_hints_reward_imp_bank(hints, weak_answer, min(self.to_explore_reward.get(refined_answer.final_answer, [])) - min(self.to_explore_reward.get(weak_answer, [])), refined_answer.final_answer)

        return hints_list, answers_list, self.to_explore, self.to_explore_reward, self.hints_bank, self.history_bank, self.hints_reward_imp_bank, self.fathers, self.childs, self.ucb_bank

    def check_termination(self, current_answer: str, iteration: int) -> bool:
        # 正解チェック
        if check(self.ground_truth, current_answer):
            return True

        # 早期停止
        if iteration > 1:
            current_reward = self.to_explore_reward[current_answer][-1]
            previous_reward = self.to_explore_reward[self.fathers[current_answer]][-1]
            if (current_reward - previous_reward) / abs(previous_reward) < self.early_stop_threshold:
                return True

        # 探索深度制限
        if len(self.fathers) > self.max_depth:
            return True

        # 言語モデルのロジットに基づく終了条件
        # 注: この部分は使用する言語モデルの仕様に応じて調整が必要です
        # logits = self.get_logits(current_answer)
        # if max(logits) > self.logits_threshold:
        #     return True

        return False

    async def get_weak_answer(self) -> Tuple[str, List[str]]:
        query = f'Question: {self.query}\nThe response should begin with [reasoning process]...[Verification]... and end with {self.ans_format}\nLet\'s think step by step.'
        return await generate(query, timeout=90)

# メイン処理
async def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
    query = example['question']
    ground_truth = example['answer']
    ans_format = '"[Final Answer] The answer is [answer] \n#### [answer]"'

    mctsr = MCTSr(query, ground_truth, max_iter=16, ans_format=ans_format)
    hints_list, answers_list, to_explore, to_explore_reward, hints_bank, history_bank, hints_reward_imp_bank, fathers, childs, ucb_bank = await mctsr.main_loop()

    return {
        'query': query,
        'ground_truth': ground_truth,
        'hints_list': [hint.model_dump() for hint in hints_list],
        'answers_list': [answer.model_dump() for answer in answers_list],
        'to_explore': to_explore,
        'to_explore_reward': to_explore_reward,
        'hints_bank': hints_bank,
        'history_bank': history_bank,
        'hints_reward_imp_bank': hints_reward_imp_bank,
        'fathers': fathers,
        'childs': childs,
        'ucb_bank': ucb_bank,
    }

# メイン実行
async def main():
    # データセットの読み込みと処理
    # dataset = load_dataset(...)
    # この部分は実際のデータセットに合わせて調整する必要があります
    dataset = [
        {"question": "2 + 2 = ?", "answer": "4"},
        {"question": "What is the capital of France?", "answer": "Paris"},
        # ... 他の例を追加 ...
    ]
    
    processed_data = []
    for example in dataset:
        result = await process_example(example)
        processed_data.append(result)
    
    # 結果の保存（例：JSON形式で）
    import json
    with open('processed_data.json', 'w') as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"処理が完了しました。{len(processed_data)}個の例が処理されました。")

if __name__ == '__main__':
    asyncio.run(main())