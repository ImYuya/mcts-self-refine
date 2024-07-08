from dotenv import load_dotenv
import os
from note_client import Note

# .envファイルから環境変数を読み込む
load_dotenv(override=True)

# 環境変数から値を取得
EMAIL = os.getenv('EMAIL')
PASSWORD = os.getenv('PASSWORD')
USER_ID = os.getenv('USER_ID')

TITLE = 'Monte Carlo Tree Search Self-Improvement (MCTSr) Algorithm: 革新的な数学的推論の最前線'
CONTENT_PATH = './docs/note/mctsr-algorithm-blog-post.md'
TAG_LIST = ['mctsr', 'algorithm']

# > If an image is specified, the index number is entered; if not, no description is given.
# INDEX = 0

# > True if the article is to be published, False if the article is to be saved as a draft; if not specified, the article is saved as a draft.
# POST_SETTING = True

# > True if the execution screen is not displayed, False if it is displayed, or not displayed if not specified.
# HEADLESS = False

# To specify the above three options, add them to the function arguments.

note = Note(email=EMAIL, password=PASSWORD, user_id=USER_ID)
print(note.create_article(title=TITLE, file_name=CONTENT_PATH, input_tag_list=TAG_LIST, image_index=None))

## If successful(Public).
# {'run':'success','title':'Sample','file_path':'content.txt','tag_list':['sample_tag'],'post_setting':'Public','post_url':'https://note.com/USER_ID/n/abc123'}

## If successful(Draft).
# {'run':'success','title':'Sample','file_path':'content.txt','tag_list':['sample_tag'],'post_setting':'Draft'}

## If unsuccessful.
# 'Required data is missing.'