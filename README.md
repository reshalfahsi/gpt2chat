# **GPT2Chat: Creating a GPT-2-Based Chatbot with Human Preferences**


<div align="center">
    <a href="https://colab.research.google.com/github/reshalfahsi/gpt2chat/blob/master/Creating_a_GPT_2_Based_Chatbot_with_Human_Preferences.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="colab"></a>
    <br />
</div>



## **Abstract**

In this project, a conversational chatbot named GPT2Chat, based on the GPT-2 language model, is developed. To enable the model to engage in meaningful dialogues, first, instruction fine-tuning is performed using the OpenAssistant Conversations Dataset (OASST1) and the Alpaca dataset. The LLaMA3 instruction template is adopted and adapted by incorporating special tokens ``<|start_context|>`` and ``<|end_context|>`` to encapsulate conversation history, thereby providing contextual awareness to the model. Subsequently, the ORPO method is employed for preference alignment, utilizing the ``trl-lib/ultrafeedback_binarized dataset`` to refine the model's responses based on human feedback. The resulting chatbot demonstrates decent conversational capabilities, leveraging both fine-tuning and preference learning techniques.



## **Introduction**

In this project, a conversational chatbot called GPT2Chat, based on the GPT-2 language model [1], is developed. The goal was to create a system that can hold natural and coherent conversations with users. To make this happen, a two-step process is used: instruction fine-tuning and preference alignment, tailoring the pre-trained GPT-2 model for dialogue generation.


## **Background**

**GPT-2** [1], created by OpenAI, is a powerful transformer-based [2] language model great at generating human-like text. However, it’s not inherently built for conversations. To adapt it for chatting, it is fine-tuned with specific datasets and aligned its responses with human preferences using advanced techniques.

- **Instruction Fine-Tuning**: This step trains the model on conversational or instruction-based datasets to improve its ability to respond to user inputs.
- **Preference Alignment**: This refines the model further by incorporating human feedback, ensuring responses are not just relevant but also user-friendly.


## **Methodology**

### **Instruction Fine-Tuning**

GPT-2 is fine-tuned using two datasets:

- **OpenAssistant Conversations Dataset (OASST1)**: A dataset full of human-AI conversations, perfect for teaching dialogue skills [3].
- **Alpaca Dataset**: Built by Stanford, this focuses on instruction-following, helping the model respond accurately to prompts [4].

To keep the chatbot aware of the conversation flow, the LLaMA3 instruction template [5] is utilized and modified. The ``<|start_context|>`` and ``<|end_context|>`` tokens are added to wrap the past conversation history or any extra context. This way, the model knows what’s been said before and can reply accordingly.

The fine-tuning process trained the model to predict the next response based on this context and the user’s latest input.


### **Preference Alignment with ORPO**

Next, the ORPO method [6] is used for preference alignment. ORPO, or Odds Ratio Preference Optimization, is a technique that improves the model’s responses by learning from human feedback. It trains the model to favor responses that people prefer while avoiding less desirable ones, using a dataset of “chosen” (good) and “rejected” (bad) response pairs. In this project, the ``trl-lib/ultrafeedback_binarized dataset``, which provides these pairs, is used to make GPT2Chat’s replies more helpful and engaging. Unlike other methods, ORPO combines this preference learning with instruction tuning in one step, making it efficient for fine-tuning GPT-2.

- **Dataset Synergy**: The ``trl-lib/ultrafeedback_binarized`` dataset contains preference pairs (chosen response, rejected response), which ORPO leverages to refine GPT2Chat’s outputs. This complements earlier SFT with OASST1 (conversational data) and Alpaca (instruction-following data), as ORPO builds on the model’s ability to generate coherent responses while aligning them with human preferences.
- **Context Tokens**: The use of ``<|start_context|>`` and ``<|end_context|>`` tokens to encode conversation history aligns well with ORPO, as the method can incorporate contextual inputs in the prompt. ORPO’s loss function evaluates responses based on the full input (prompt + context), ensuring GPT2Chat remains contextually aware during preference alignment.
- **Efficiency**: Since GPT-2 is a smaller model, ORPO’s single-step process and lack of a reference model make it computationally feasible for this project, likely reducing training time and resource needs compared to RLHF [7] or DPO [8].
- **Outcome**: ORPO enhances GPT2Chat’s ability to generate user-preferred responses, such as more helpful, accurate, or engaging replies. For example, if a user asks, “What’s the best way to learn Python?” ORPO ensures GPT2Chat favors a detailed, practical response over a vague or incorrect one, based on the preference data.

This step ensures the chatbot doesn’t just make sense—it also generates replies people actually like.


## **Implementation**

Here’s how it is put all together:

1. **Data Preparation**: The OASST1 and Alpaca datasets are processed to fit the modified template, wrapping conversation history in context tokens.
2. **Fine-Tuning**: Using frameworks (i.e., PyTorch [9], HuggingFace [10], and PyTorch Lightning [11]), GPT-2 is fine-tuned to optimize its dialogue performance.
3. **Preference Alignment**: ORPO is applied with the ``trl-lib/ultrafeedback_binarized`` dataset, using feedback to reward better responses.
4. **Testing**: The chatbot’s ability to chat naturally is checked.


## **Results**

### **Quantitative Evaluation**

The following metrics measure the performance of GPT2Chat:

| Model         | Params | FLOPs/token | Token/sec. | PPL ↓ |
|---------------|------------|-------------|------------|---------------|
| **GPT2Chat (w/o ORPO)**  | 125M | 125M | 90.214 | 15.712 |
| **GPT2Chat**  | 125M | 125M | 90.214 | **12.889** |

- **Perplexity**: Calculated on the OASST1 validation set.
- **Inference Speed**: Measured in token/sec.
- **Compute Overhead**: Measured in FLOPs/token.


### **Qualitative Evaluation**

#### GPT2Chat with and without ORPO

Examples below demonstrate the difference between the responses of GPT2Chat without any preference alignment and GPT2Chat with ORPO.

- **Example 1:**

<table>
    <tr>
        <td>
            Model
        </td>
        <td>
            Prompt
        </td>
        <td>
            Response
        </td>
    </tr>
    <tr>
        <td>
            GPT2Chat w/o ORPO
        </td>
        <td>
            What’s the best way to learn Python?
        </td>
        <td>
            The best way to learn Python is to use it as often as you can, read beginner-friendly books and articles, watch tutorials and examples, and participate in online communities.
        </td>
    </tr>
    <tr>
        <td>
            GPT2Chat
        </td>
        <td>
            What’s the best way to learn Python?
        </td>
        <td>
            The best way to learn Python is to start with a basic understanding of the language, such as the basic syntax, rules, and syntax. You should also make sure to have a strong understanding of the language's history, including the core concepts, frameworks, and frameworks. Additionally, you should have a good understanding of the ecosystem and its ecosystem, as well as practicing your understanding through practice.
        </td>
    </tr>
</table>

- **Example 2:**

<table>
    <tr>
        <td>
            Model
        </td>
        <td>
            Prompt
        </td>
        <td>
            Response
        </td>
    </tr>
    <tr>
        <td>
            GPT2Chat w/o ORPO
        </td>
        <td>
            How can I improve my English skills?
        </td>
        <td>
            There are many ways to improve your English skills, such as practicing speaking with native speakers, reading English books and articles, watching English movies and TV shows, listening to English podcasts and speaking with native English speakers, and practicing the English vocabulary. Additionally, you should study for English exam and spelling, as well as writing and editing, to improve your English skills.
        </td>
    </tr>
    <tr>
        <td>
            GPT2Chat
        </td>
        <td>
            How can I improve my English skills?
        </td>
        <td>
            To improve your English skills, try the following:

            1. Learn the language: Start by learning the language from a book, magazine, or online newspaper, and gradually increase your English skills as you become more familiar with the language.

            2. Use an English tutor: An English tutor can help you improve your English skills by providing practical advice and guidance. They can help you learn basic vocabulary, grammar, and basic vocabulary, as well as provide a support system to help you practice in English.

            3. Study vocabulary: Spend time studying in English, such as practicing with English
        </td>
    </tr>
</table>

- **Example 3:**

<table>
    <tr>
        <td>
            Model
        </td>
        <td>
            Prompt
        </td>
        <td>
            Response
        </td>
    </tr>
    <tr>
        <td>
            GPT2Chat w/o ORPO
        </td>
        <td>
            How can we maintain our well-being?
        </td>
        <td>
            We can maintain our well-being by taking care of ourselves and our families, eating nutritious meals, getting enough sleep, engaging in physical activity, engaging in creative activities, and taking care of our physical health.
        </td>
    </tr>
    <tr>
        <td>
            GPT2Chat
        </td>
        <td>
            How can we maintain our well-being?
        </td>
        <td>
            Here are some ways to maintain your well-being:

            1. Reduce stress: Find ways to reduce stress, such as spending time in nature or engaging in active exercise.

            2. Reduce stress: Find ways to reduce stress, such as practicing relaxation techniques, practicing relaxation techniques, or taking time for self-care.

            3. Exercise regularly: Regular exercise can help improve your overall health and well-being, as it can reduce the risk of developing chronic illnesses such as heart disease, stroke, and other chronic conditions.

            4. Avoid stress: Avoiding stress and
        </td>
    </tr>
</table>


#### **Live on Colab**

<p align="center"> <img src="https://github.com/reshalfahsi/gpt2chat/blob/master/assets/gpt2chat.png" alt="gpt2chat" > <br /> A sample of conversation with GPT2Chat on Colab. </p>

#### **Telegram Bot**

<table>
    <tr>
        <td> 
            <img src="https://github.com/reshalfahsi/gpt2chat/blob/master/assets/gpt2chat-00.gif" alt="gpt2chat-00" > 
        </td>
        <td> 
            <img src="https://github.com/reshalfahsi/gpt2chat/blob/master/assets/gpt2chat-01.gif" alt="gpt2chat-01" > 
        </td>
        <td> 
            <img src="https://github.com/reshalfahsi/gpt2chat/blob/master/assets/gpt2chat-02.gif" alt="gpt2chat-02" > 
        </td>
    </tr>
</table>


<div align="center">

GPT2Chat conversational ability via Telegram Bot. With LangChain [12], the conversation flow becomes manageable.

</div>


## **References**

1. [A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever, and others, "Language models are unsupervised multitask learners," OpenAI blog, vol. 1, no. 8, p. 9, 2019.](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
2. [A. Vaswani et al., "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017.](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
3. [A. Köpf et al., "Openassistant conversations-democratizing large language model alignment," in *Advances in Neural Information Processing Systems*, vol. 36, pp. 47669–47681, 2023.](https://proceedings.neurips.cc/paper_files/paper/2023/file/949f0f8f32267d297c2d4e3ee10a2e7e-Paper-Datasets_and_Benchmarks.pdf)
4. R. Taori et al., "Alpaca: A strong, replicable instruction-following model," *Stanford Center for Research on Foundation Models*, vol. 3, no. 6, pp. 7, 2023. [Online]. Available: [https://crfm.stanford.edu/2023/03/13/alpaca.html](https://crfm.stanford.edu/2023/03/13/alpaca.html)
5. [A. Grattafiori et al., "The llama 3 herd of models," arXiv preprint arXiv:2407.21783, 2024.](https://arxiv.org/pdf/2407.21783)
6. [J. Hong, N. Lee, and J. Thorne, "Orpo: Monolithic preference optimization without reference model," arXiv preprint arXiv:2403.07691, 2024.](https://arxiv.org/pdf/2403.07691)
7. [L. Ouyang et al., "Training language models to follow instructions with human feedback," in *Advances in Neural Information Processing Systems*, vol. 35, pp. 27730–27744, 2022.](https://proceedings.neurips.cc/paper_files/paper/2022/file/b1efde53be364a73914f58805a001731-Paper-Conference.pdf)
8. [R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, "Direct preference optimization: Your language model is secretly a reward model," in *Advances in Neural Information Processing Systems*, vol. 36, pp. 53728–53741, 2023.](https://papers.nips.cc/paper_files/paper/2023/file/a85b405ed65c6477a4fe8302b5e06ce7-Paper-Conference.pdf)
9. [A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, pp. 8026–8037, 2019.](https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Paper.pdf)
10. [T. Wolf et al., "Huggingface's transformers: State-of-the-art natural language processing," *arXiv preprint arXiv:1910.03771*, 2019.](https://arxiv.org/pdf/1910.03771)
11. [W. Falcon and The PyTorch Lightning team, "PyTorch Lightning," 2019.](https://github.com/Lightning-AI/pytorch-lightning)
12. H. Chase, "LangChain," GitHub, Oct. 2022. [Online]. Available: [https://github.com/langchain-ai/langchain](https://github.com/langchain-ai/langchain)
