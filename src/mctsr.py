import math
import re
from typing import List, Dict, Tuple, Any, Optional
from functools import lru_cache
import asyncio
import instructor
from openai import OpenAI
import time
import ollama
import numpy as np
from scipy.spatial.distance import cosine

LLM_MODEL = 'gemma2'
EMBEDDING_MODEL = 'mxbai-embed-large'

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

async def generate(prompt: str, history: List[str] = [], timeout: int = 150) -> Tuple[str, List[str]]:
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
        response = await asyncio.wait_for(
            asyncio.to_thread(
                llm.client.chat.completions.create,
                model=llm.model,
                messages=messages,
                response_model=str
            ),
            timeout=timeout
        )
        
        updated_history = history + [prompt, response]
        
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
    return embedding[0]['embedding']

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

    async def sampling_reward(self, answer: str) -> None:
        if answer not in self.to_explore_reward:
            self.to_explore_reward[answer] = []
        reward = await self.cal_reward(self.query, answer)
        self.to_explore_reward[answer].append(reward)

    async def cal_reward(self, question: str, ans: str) -> float:
        # 報酬計算のロジックを実装
        # この例では、LLMを使用して報酬を計算します
        prompt = f"Question: {question}\nAnswer: {ans}\nAnalyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score!\nOutput a score between [-100,+100]."
        response, _ = await generate(prompt)
        # スコアを抽出（この部分は実際の応答形式に応じて調整が必要）
        score = float(re.search(r'-?\d+', response).group())
        return score

    def add_to_hints_bank(self, hints: str, weak_answer: str) -> None:
        if weak_answer not in self.hints_bank:
            self.hints_bank[weak_answer] = []
        self.hints_bank[weak_answer].append(hints)

    def add_to_childs(self, father: str, child: str) -> None:
        if father not in self.childs:
            self.childs[father] = []
        self.childs[father].append(child)

    def add_to_hints_reward_imp_bank(self, hints: str, weak_answer: str, reward: float, answer: str) -> None:
        if weak_answer not in self.hints_reward_imp_bank:
            self.hints_reward_imp_bank[weak_answer] = []
        self.hints_reward_imp_bank[weak_answer].append((hints, reward, answer))

    def filter_mature_node(self, max_expand: int = 3) -> List[str]:
        filterd_to_explore = []
        avg_reward = {node: (min(self.to_explore_reward[node]) + sum(self.to_explore_reward[node]) / len(self.to_explore_reward[node])) / 2 for node in self.to_explore}

        for node in self.to_explore:
            if len(self.childs.get(node, [])) < max_expand or max([avg_reward.get(child, -999) for child in self.childs.get(node, [])]) < avg_reward.get(node, -999):
                filterd_to_explore.append(node)
        
        return filterd_to_explore

    def get_best_explore_from_ucb(self, to_explore: List[str]) -> str:
        best_node = None
        highest_ucb = float('-inf')
        
        for node in to_explore:
            ucb_value = self.ucb_bank.get(node, float('-inf'))
            if ucb_value > highest_ucb:
                highest_ucb = ucb_value
                best_node = node
                
        return best_node

    def compute_ucb(self, r_c: float, N_n: int, N_c: int, C: float = 1.4) -> float:
        return r_c + C * math.sqrt(math.log(N_n + 1) / (N_c + 1e-5))

    def update_ucb(self, C: float = 1.4) -> None:
        visit_count = {node: len(self.to_explore_reward[node]) for node in self.to_explore}
        avg_reward = {node: (min(self.to_explore_reward[node]) + sum(self.to_explore_reward[node]) / len(self.to_explore_reward[node])) / 2 for node in self.to_explore}

        leaves = set(self.to_explore) - set(self.fathers.values())
        
        for leaf in leaves:
            self.ucb_bank[leaf] = self.compute_ucb(avg_reward[leaf], len(self.to_explore_reward.get(self.fathers.get(leaf, None), [])), len(self.to_explore_reward.get(leaf, [])), C)
        
        nodes_to_update = list(leaves)
        while nodes_to_update:
            new_nodes_to_update = set()
            for node in nodes_to_update:
                father = self.fathers.get(node)
                if father is not None:
                    if father not in self.ucb_bank:
                        new_nodes_to_update.add(father)
                    if father in self.ucb_bank:
                        child_reward = [avg_reward[child] for child in self.childs[father]]
                        father_reward = (avg_reward[father] + max(child_reward)) / 2
                        self.ucb_bank[father] = self.compute_ucb(father_reward, len(self.to_explore_reward.get(self.fathers.get(father, None), [])), len(self.to_explore_reward.get(father, [])), C)
            nodes_to_update = list(new_nodes_to_update)

    async def step(self, weak_answer: str) -> Tuple[str, str, List[str]]:
        hints, history = await self.get_weak_hints(self.query, weak_answer, history=self.history_bank[weak_answer])
        answer, history = await self.get_better_answer(self.query, weak_answer, hints, history=history)
        return hints, answer, history

    async def get_weak_hints(self, question: str, weak_answer: str, history: List[str] = []) -> Tuple[str, List[str]]:
        query = f'Question: {question}\nSince we have a weak Answer, could you provide me with a reflection or feedback to correct this answer better? Analyze this Answer Strictly and Critic, point out every flaw for every possible imperfect to minus every possible score!\nLet\'s think step by step.'
        return await generate(query, history)

    async def get_better_answer(self, question: str, weak_answer: str, hint: str, history: List[str] = []) -> Tuple[str, List[str]]:
        query = f'Question: {question}\nPlease refine your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with {self.ans_format}\nLet\'s think step by step.'
        return await generate(query, history)

    async def main_loop(self) -> Tuple[List[str], List[str], List[str], Dict[str, List[float]], Dict[str, List[str]], Dict[str, Tuple[str, ...]], Dict[str, List[Tuple[str, float, str]]], Dict[str, str], Dict[str, List[str]], Dict[str, float]]:
        weak_answer, history = await self.get_weak_answer()
        self.history_bank[weak_answer] = tuple(history)
        self.to_explore = [weak_answer]
        self.childs[weak_answer] = []
        self.fathers[weak_answer] = None
        await self.sampling_reward(weak_answer)

        hints_list = []
        answers_list = [weak_answer]

        self.update_ucb()

        for _ in range(self.max_iter):
            filtered_to_explore = self.filter_mature_node()
            weak_answer = self.get_best_explore_from_ucb(filtered_to_explore)
            await self.sampling_reward(weak_answer)

            hints, answer, history = await self.step(weak_answer)
            self.add_to_hints_bank(hints, weak_answer)
            self.history_bank[answer] = tuple(history)
            self.to_explore.append(answer)
            await self.sampling_reward(answer)
            self.fathers[answer] = weak_answer
            self.childs[answer] = []
            self.add_to_childs(weak_answer, answer)
            answers_list.append(answer)
            hints_list.append(hints)

            if check(self.ground_truth, answer):
                break

            self.update_ucb()
            self.add_to_hints_reward_imp_bank(hints, weak_answer, min(self.to_explore_reward.get(answer, [])) - min(self.to_explore_reward.get(weak_answer, [])), answer)

        return hints_list, answers_list, self.to_explore, self.to_explore_reward, self.hints_bank, self.history_bank, self.hints_reward_imp_bank, self.fathers, self.childs, self.ucb_bank

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
        'hints_list': hints_list,
        'answers_list': answers_list,
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