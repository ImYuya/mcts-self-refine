# A  Prompts in Experiment
    
## A.1 Self-Refine
    
### Get Feedback:
    
    > Analyze the given answer critically and thoroughly. Please provide detailed feedback, highlighting every flaw and imperfection. Consider all possible aspects that could be improved. Be comprehensive in your critique.
    > 
    > 
    > Let's approach this step-by-step:
    > 
    > 1. Identify major flaws in reasoning
    > 2. Point out any factual inaccuracies
    > 3. Assess the structure and clarity of the answer
    > 4. Evaluate the depth and breadth of the content
    > 5. Consider any missing key points or perspectives
    
### Get Refined Answer:
    
    > USER: Please refine the your answer according to your Reflection or Feedback. The response should begin with [reasoning process]...[Verification]... and end with end with "[Final Answer] The answer is [answer formula]"
    Let's think step by step.
    > 
    
    > Based on the feedback provided, please refine your answer. Structure your response as follows:
    > 
    > 
    > [Reasoning Process]
    > 
    > - Explain your thought process
    > - Describe how you're addressing the feedback
    > 
    > [Verification]
    > 
    > - Double-check your refined answer
    > - Ensure all critique points have been addressed
    > 
    > [Final Answer]
    > The refined answer is: [Insert your improved answer here]
    > 
    > Remember to think through each step carefully as you refine your response.
    > 
    
## A.2 Self-Reward
    
    > Question: [Insert question here]
    Initial Answer: [Insert initial answer here]
    > 
    > 
    > Task:
    > 
    > 1. Analyze this answer rigorously and critically.
    > 2. Identify and explain every flaw, no matter how small.
    > 3. Be extremely strict in your evaluation - do not hesitate to point out imperfections.
    > 4. Assign a score between -100 and +100, where:
    >     - 100 represents a completely incorrect or irrelevant answer
    >     - +100 represents a theoretically perfect answer (note: this score should rarely, if ever, be given)
    > 
    > Provide your analysis in the following format:
    > [Detailed Critique]: (List all identified flaws and areas for improvement)
    > [Score]: (Assign a numerical score between -100 and +100)
    > [Justification]: (Explain the reasoning behind your score)
    > 
    
## A.3 Dummy Answers
    
    > Instructions: When you are unable to provide a substantive or confident answer to a question, select the most appropriate response from the following list. Ensure that your choice accurately reflects your inability to answer while maintaining a professional tone.
    > 
    > 
    > Available Responses:
    > [
    > "I Don't Know",
    > "I can't understand this question.",
    > "I can't help with this question.",
    > "I don't know how to solve this question.",
    > "I don't know the answer to this question.",
    > "I don't know the answer to this question, sorry."
    > ]
    > 
    > Usage Guidelines:
    > 
    > 1. Choose the response that best fits the context of the question.
    > 2. Use these responses sparingly and only when genuinely unable to provide an answer.
    > 3. If possible, follow up with a suggestion for where the user might find the information or an offer to assist with a related topic you can address.
    > 4. Maintain consistency in your tone and level of formality when using these responses.
    > 
    > Remember: It's better to admit lack of knowledge than to provide inaccurate information.
    >