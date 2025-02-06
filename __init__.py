# Logging
from loguru import logger
logger.add('run.log', rotation="1 week", retention="4 weeks")    # Once the file is too old, it's rotated
# Install NLTK Data
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# Load environment variables
import dotenv
dotenv.load_dotenv()