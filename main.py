from flask import jsonify, request, Flask
from transformers import  DebertaTokenizer, DebertaForSequenceClassification
from transformers import TextClassificationPipeline
import regex as re

model = DebertaForSequenceClassification.from_pretrained("ImanAndrea/deberta-base-amazon-products-classifier")
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=True)

app = Flask(__name__)

def preprocess(product):
    product = product.lower()
    product = re.sub(r"[\[\(\:\/\\\)\|,\]\-\_]", "", product)
    return product

@app.route('/', methods=['GET'])
def main():
    return "main page"

@app.route('/classify',methods=['POST'])
def classify():
    data = request.json
    product = str(data.get("product", ""))
    product = preprocess(product)
    
    if product=="":
        return jsonify(msg="No sentence given"), 400

    predictions = pipe(product)[0]
    
    if predictions is None:
        return jsonify(msg="No prediction obtained"), 400
    
    predictions = sorted(predictions, key=lambda a:a["score"], reverse=True)
    
    return jsonify(predictions), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8500)