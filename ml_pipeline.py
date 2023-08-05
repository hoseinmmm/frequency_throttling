# ----------------------------- Sync with c
import os
import errno


import argparse
# Initialize parser
parser = argparse.ArgumentParser()
# Adding optional argument
parser.add_argument("Number", type=int, help="Input Integer")
# Read arguments from command line
args = parser.parse_args()


FIFO = 'mypipe'

try:
    os.mkfifo(FIFO)
except OSError as oe: 
    if oe.errno != errno.EEXIST:
        raise Exception
#--------------------------------------------------


from transformers import ViTForImageClassification
import torch

device = torch.device('cpu')#'cuda' if torch.cuda.is_available() else 'cpu')


# ---------------------  Input preperation
from PIL import Image
import requests


url="http://images.cocodataset.org/val2017/000000039769.jpg"
try:
    image = Image.open(requests.get(url, stream=True).raw)
except:
    url='https://yt3.googleusercontent.com/ytc/AGIKgqO2_c-cM4m3V9CdcUuZeTObiIL4Cd4Qh-4NNy6CXg=s900-c-k-c0x00ffffff-no-rj'
    image = Image.open(requests.get(url, stream=True).raw)

#---------------------------------------------




# ------------------------------ Model selection
model_sel = args.Number

model_name = None

pointer_inputs = []
timm_models = []
nlp_models = []

if (model_sel == 1):
    print("model_sel is " + str(model_sel) )
    model_name = 'google/vit-base-patch16-224'
    model = ViTForImageClassification.from_pretrained(model_name)
    model.to(device)
    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained(model_name)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    print(pixel_values.shape)
    loop = 50
        
    
elif(model_sel == 2):
    print("model_sel is " + str(model_sel) )
    model_name = 'facebook/deit-tiny-patch16-224'
    model = ViTForImageClassification.from_pretrained(model_name)
    model.to(device)
    from transformers import ViTImageProcessor
    processor = ViTImageProcessor.from_pretrained(model_name)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    print(pixel_values.shape)
    loop = 300

elif(model_sel == 3):
    print("model_sel is " + str(model_sel) )
    model_name = 'google/vit-hybrid-base-bit-384'
    from transformers import ViTHybridForImageClassification, ViTHybridImageProcessor
    model = ViTHybridForImageClassification.from_pretrained(model_name)
    model.to(device)
    processor = ViTHybridImageProcessor.from_pretrained(model_name)
    inputs = processor(images=image, return_tensors="pt").to(device)
    pixel_values = inputs.pixel_values
    print(pixel_values.shape)
    loop = 40

elif(model_sel == 4):
    print("model_sel is " + str(model_sel) )
    model_name = 'vit_base_patch16_224.augreg_in21k'
    import timm
    model = timm.create_model(model_name, pretrained=True)
    model = model.eval()
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)
    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 40
    timm_models.append(model_sel)

elif(model_sel == 5):
    print("model_sel is " + str(model_sel) )
    model_name = 'nateraw/vit-age-classifier'
    from transformers import ViTImageProcessor, ViTForImageClassification
    model = ViTForImageClassification.from_pretrained(model_name)
    transforms = ViTImageProcessor.from_pretrained(model_name)
    from io import BytesIO
    r = requests.get('https://github.com/dchen236/FairFace/blob/master/detected_faces/race_Asian_face0.jpg?raw=true')
    im = Image.open(BytesIO(r.content))
    inputs = transforms(im, return_tensors='pt')
    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 20
    pointer_inputs.append(model_sel)

elif(model_sel == 6):
    print("Nop operation")

elif(model_sel == 7):
    print("model_sel is " + str(model_sel) )
    model_name = 'apple/mobilevit-small'
    from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
    feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
    model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 40
    pointer_inputs.append(model_sel)


elif(model_sel == 8):
    print("model_sel is " + str(model_sel) )
    model_name = 'microsoft/resnet-50'

    from transformers import AutoImageProcessor, ResNetForImageClassification

    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    inputs = processor(image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 40
    pointer_inputs.append(model_sel)


elif(model_sel == 9):
    print("model_sel is " + str(model_sel) )
    model_name = 'facebook/convnext-large-224'

    from transformers import ConvNextImageProcessor, ConvNextForImageClassification

    processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-large-224")
    model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224")

    inputs = processor(image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 15
    pointer_inputs.append(model_sel)


elif(model_sel == 10):
    print("model_sel is " + str(model_sel) )
    model_name = 'microsoft/resnet-18'

    from transformers import AutoFeatureExtractor, ResNetForImageClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

    inputs = feature_extractor(image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 60
    pointer_inputs.append(model_sel)

elif(model_sel == 11):
    print("model_sel is " + str(model_sel) )
    model_name = 'microsoft/resnet-101'

    from transformers import AutoFeatureExtractor, ResNetForImageClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-101")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-101")

    inputs = feature_extractor(image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 30
    pointer_inputs.append(model_sel)

elif(model_sel == 12):
    print("model_sel is " + str(model_sel) )
    model_name = 'microsoft/resnet-152'

    from transformers import AutoFeatureExtractor, ResNetForImageClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-152")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-152")

    inputs = feature_extractor(image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 30
    pointer_inputs.append(model_sel)

elif(model_sel == 13):
    print("model_sel is " + str(model_sel) )
    model_name = 'google/mobilenet_v2_1.4_224'

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.4_224")
    model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_1.4_224")

    inputs = preprocessor(images=image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 70
    pointer_inputs.append(model_sel)

elif(model_sel == 14):
    print("model_sel is " + str(model_sel) )
    model_name = 'google/mobilenet_v2_0.75_160'

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_0.75_160")
    model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v2_0.75_160")

    inputs = preprocessor(images=image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 200
    pointer_inputs.append(model_sel)

elif(model_sel == 15):
    print("model_sel is " + str(model_sel) )
    model_name = 'Matthijs/mobilenet_v1_1.0_224'

    from transformers import MobileNetV1FeatureExtractor, MobileNetV1ForImageClassification

    feature_extractor = MobileNetV1FeatureExtractor.from_pretrained("Matthijs/mobilenet_v1_1.0_224")
    model = MobileNetV1ForImageClassification.from_pretrained("Matthijs/mobilenet_v1_1.0_224")

    inputs = feature_extractor(images=image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 200
    pointer_inputs.append(model_sel)

elif(model_sel == 16):
    print("model_sel is " + str(model_sel) )
    model_name = 'inception_v3.tf_adv_in1k'
    import timm

    model = timm.create_model('inception_v3.tf_adv_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 30
    timm_models.append(model_sel)


elif(model_sel == 17):
    print("model_sel is " + str(model_sel) )
    model_name = 'inception_v3.gluon_in1k'
    import timm

    model = timm.create_model('inception_v3.gluon_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 30
    timm_models.append(model_sel)


elif(model_sel == 18):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    # Download and load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load the pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # List of strings to classify
    #strings = ["I love this movie!", "This food tastes awful.", "What a beautiful day!"]
    strings = ["I love this movie and what a beautiful day that I want to go outside and play some nice music so my friend enojys and I feel very good in his point of view."]
    #strings = ["Nice day!"]
    loop = 6
    nlp_models.append(model_sel)


elif(model_sel == 19):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = "textattack/bert-base-uncased-QNLI"

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    # Download and load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load the pipeline
    nlp = pipeline('text-classification', model=model, tokenizer=tokenizer)

    # List of strings to classify
    # For QNLI, the format is "[CLS] question [SEP] context [SEP]"
    strings = ["[CLS] Who won the world series in 2020? [SEP] The Los Angeles Dodgers won the World Series in 2020. [SEP]", 
            "[CLS] What is the capital of France? [SEP] Paris is the capital of France. [SEP]"]
    loop = 5
    nlp_models.append(model_sel)

elif(model_sel == 20):
    print("model_sel is " + str(model_sel) )
    # Specify the model


    model_name = "facebook/convnext-large-224-22k-1k"
    from transformers import ConvNextFeatureExtractor, ConvNextForImageClassification
    feature_extractor = ConvNextFeatureExtractor.from_pretrained("facebook/convnext-large-224-22k-1k")
    model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224-22k-1k")


    inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = inputs

    loop = 25
    pointer_inputs.append(model_sel)

elif(model_sel == 21):
    print("model_sel is " + str(model_sel) )

    
    # Specify the model
    #model_name = "google/vit-base-patch16-384"
    model_name = 'ahishamm/vit-base-isic-patch-32'

    from transformers import ViTFeatureExtractor, ViTForImageClassification

    feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    inputs = feature_extractor(images=image, return_tensors="pt")
    

    pixel_values = inputs

    loop = 15
    pointer_inputs.append(model_sel)

elif(model_sel == 22):

    print("model_sel is " + str(model_sel) )
    model_name = 'microsoft/resnet-18'

    from transformers import AutoFeatureExtractor, ResNetForImageClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")

    inputs = feature_extractor(image, return_tensors="pt")

    pixel_values = inputs
    #print(pixel_values.shape)
    loop = 60
    pointer_inputs.append(model_sel)

elif(model_sel == 23):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = "microsoft/resnet-34"

    from transformers import AutoFeatureExtractor, ResNetForImageClassification

    feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-34")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-34")

    inputs = feature_extractor(image, return_tensors="pt")

    pixel_values = inputs

    loop = 60
    pointer_inputs.append(model_sel)

elif(model_sel == 24):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = "google/mobilenet_v1_0.75_192"

    from transformers import AutoImageProcessor, AutoModelForImageClassification

    preprocessor = AutoImageProcessor.from_pretrained("google/mobilenet_v1_0.75_192")
    model = AutoModelForImageClassification.from_pretrained("google/mobilenet_v1_0.75_192")

    inputs = preprocessor(images=image, return_tensors="pt")

    pixel_values = inputs

    loop = 250
    pointer_inputs.append(model_sel)

elif(model_sel == 25):
    print("model_sel is " + str(model_sel) )
    # Specify the model


    model_name = "inception_resnet_v2.tf_ens_adv_in1k"
    import timm
    model = timm.create_model('inception_resnet_v2.tf_ens_adv_in1k', pretrained=True)
    model = model.eval()
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 35
    timm_models.append(model_sel)

elif(model_sel == 26):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = "distilbert-base-uncased"

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    # Download and load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load the pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # List of strings to classify
    #strings = ["I love this movie!", "This food tastes awful.", "What a beautiful day!"]
    strings = ["I love this movie and what a beautiful day that I want to go outside and play some nice music so my friend enojys and I feel very good in his point of view."]
    #strings = ["Nice day!"]
    loop = 6
    nlp_models.append(model_sel)

elif(model_sel == 27):
    print("model_sel is " + str(model_sel) )
    # Specify the model

    model_name = "resnet50.tv_in1k"
    import timm
    model = timm.create_model('resnet50.tv_in1k', pretrained=True)
    model = model.eval()
    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 40
    timm_models.append(model_sel)

elif(model_sel == 28):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = 'facebook/convnextv2-tiny-22k-384'

    from transformers import AutoImageProcessor, ConvNextV2ForImageClassification

    preprocessor = AutoImageProcessor.from_pretrained("facebook/convnextv2-tiny-22k-384")
    model = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-22k-384")

    inputs = preprocessor(image, return_tensors="pt")
    pixel_values = inputs

    loop = 20
    pointer_inputs.append(model_sel)

elif(model_sel == 29):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = "bhadresh-savani/bert-base-go-emotion"

    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    # Download and load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Load the pipeline
    nlp = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # List of strings to classify
    #strings = ["I love this movie!", "This food tastes awful.", "What a beautiful day!"]
    strings = ["I love this movie and what a beautiful day that I want to go outside and play some nice music so my friend enojys and I feel very good in his point of view."]
    #strings = ["Nice day!"]
    loop = 6
    nlp_models.append(model_sel)


#--------------------

elif(model_sel == 30):
    print("model_sel is " + str(model_sel) )
    # Specify the model

    model_name = "vgg11.tv_in1k"
    import timm
    model = timm.create_model('vgg11.tv_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 10
    timm_models.append(model_sel)

elif(model_sel == 31):
    print("model_sel is " + str(model_sel) )
    # Specify the model

    model_name = "vgg13.tv_in1k"
    import timm
    model = timm.create_model('vgg13.tv_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 10
    timm_models.append(model_sel)

elif(model_sel == 32):
    print("model_sel is " + str(model_sel) )
    # Specify the model

    model_name = "vgg19.tv_in1k"
    import timm
    model = timm.create_model('vgg19.tv_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 10
    timm_models.append(model_sel)



elif(model_sel == 33):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = 'nvidia/mit-b0'

    from transformers import SegformerFeatureExtractor, SegformerForImageClassification

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b0")
    model = SegformerForImageClassification.from_pretrained("nvidia/mit-b0")

    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs

    loop = 10
    pointer_inputs.append(model_sel)

elif(model_sel == 34):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = 'nvidia/mit-b2'

    from transformers import SegformerFeatureExtractor, SegformerForImageClassification

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b2")
    model = SegformerForImageClassification.from_pretrained("nvidia/mit-b2")

    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs

    loop = 10
    pointer_inputs.append(model_sel)

elif(model_sel == 35):
    print("model_sel is " + str(model_sel) )
    # Specify the model

    model_name = "vgg16.tv_in1k"
    import timm
    model = timm.create_model('vgg16.tv_in1k', pretrained=True)
    model = model.eval()

    # get model specific transforms (normalization, resize)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)


    pixel_values = transforms(image).unsqueeze(0)
    print(pixel_values.shape)
    loop = 10
    timm_models.append(model_sel)


elif(model_sel == 36):
    print("model_sel is " + str(model_sel) )
    # Specify the model
    model_name = 'nvidia/mit-b1'

    from transformers import SegformerFeatureExtractor, SegformerForImageClassification

    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b1")
    model = SegformerForImageClassification.from_pretrained("nvidia/mit-b1")

    inputs = feature_extractor(images=image, return_tensors="pt")
    pixel_values = inputs

    loop = 10
    pointer_inputs.append(model_sel)



'''
# models are downloaded at ~/.cache/huggingface/hub
'''

print("Model is loaded!")

#------------------------------------------------



if model_sel in nlp_models:
    for i in range(loop):
        print(len(strings))
        strings = strings + strings
    print("string is ready. lenght: " + str(len(strings)))




#--------------------------- Send signal_ready for inference to C
if True:
    with open(FIFO, 'w') as f:
        f.write('x\n')
#------------------------------------------



# ---------------------- Inference


if model_sel==6:
    import time
    for i in range(7000):
        time.sleep(1000 / 1000000)
    print("Done")

elif model_sel in nlp_models:

    # Perform text classification
    results = nlp(strings)
    print("*************************************************\nDone")
    # Print out results
    for string, result in zip(strings, results):
        print(f"'{string}' is {result['label']} with a score of {round(result['score'], 4)}")


else:
    
    outputs_l = []
    for i in range(loop):
        
        with torch.no_grad():
            #for models get input as **
            if model_sel in pointer_inputs:
                outputs = model(**pixel_values)
            else:
                outputs = model(pixel_values)

        outputs_l.append(outputs)

    print("*************************************************\nDone")
    prediction = []
    for outputs in outputs_l:
        # for models using timm
        if model_sel not in timm_models:
            logits = outputs.logits
            prediction.append(logits.argmax(-1))


    for i in range(loop): 
        if model_sel not in timm_models:
            print("Predicted class:", model.config.id2label[prediction[i].item()])
    print("Done")
#--------------------------------


