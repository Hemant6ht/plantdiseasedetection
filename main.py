from __future__ import division, print_function
from flask import Flask,render_template,request,session,redirect,Markup
from flask_sqlalchemy import SQLAlchemy
import json
import os
from werkzeug.utils import secure_filename
#from here everything is added from ml

# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
#from keras.preprocessing import image
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
import cv2 
from matplotlib import pyplot as plt 
from PIL import Image
from scipy.signal import convolve2d


#ml imports ends here

with open('parameters.json','r') as c:
    param=json.load(c)["params"]

local = True
log = False
username=False

app = Flask(__name__)

MODEL_PATH = r'D:\All_projects\plantdiseasedetection\models\AlexNetModel.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()     
# Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')
#names of the classfrom keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing.image import ImageDataGenerator
#valid_datagen = ImageDataGenerator(rescale=1./255)
#valid_set = valid_datagen.flow_from_directory(r"C:\Users\KIIT\Desktop\plant_disease\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)/valid",target_size=(224, 224),batch_size=128,class_mode='categorical')
#class_dict = valid_set.class_indices
#li = list(class_dict.keys())
li= ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry(including_sour)__Powdery_mildew', 'Cherry(including_sour)__healthy', 'Corn(maize)__Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)__Common_rust', 'Corn_(maize)__Northern_Leaf_Blight', 'Corn(maize)__healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
print(li)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


#convert to string result file
def convert(s):
    if(s=='Apple_Apple_scab'):
        return('<h1><font color="RED">Disease</font>: Apple Scab <h1><h3><font color="GREEN">Remedy</h3></font><p>When watering your apple trees, avoid getting foliage wet. \n Apply dolomitic lime in the fall, after leaf drop, to increase pH.Symptoms on fruit are similar to those found on leaves.</p>')
    elif(s=='Apple_Black_rot'):
        return('<h1><font color="RED">Disease</font>: Apple Black rot<h1><h3><font color="GREEN">Remedy</h3></font><p>Although black rot cankers, fruit infection, and frogeye leaf spot can cause serious losses on apples and crabapples.\n Remove the cankers by pruning at least 15 inches below the end and burn or bury them.\nFungicides are also an option to keep these diseases at bay.</p>')
    elif(s=='Apple_Cedar_apple_rust'):
        return('<h1><font color="RED">Disease</font>: Apple Rust<h1><h3><font color="GREEN">Remedy</h3></font><p>Sever affected branches 2 inches from the gall with bypass pruners or long-reach pruners using a clean straight cut.\nAllow the healthy portion of the branch to remain intact and attached to the tree.\nApply contact fungicide to trees in close proximity to the infected cedar according to the manufacturer’s guidelines.Look for the active ingredient potassium bicarbonate, approved in California for contact fungicide. Spray the trees, including trunk, branches and foliage, when you see yellow spots on the leaves, which often occurs in mid-April.</p>')
    elif(s=='Apple_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Apple Leaf<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\n Just keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Blueberry_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Blueberry Leaf<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\n Just keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Cherry_(including_sour)_Powdery_mildew'):
        return('<h1><font color="RED">Disease</font>: Cherry(including_sour) Powdery Mildew<h1><h3><font color="GREEN">Remedy</h3></font><p>Plant resistant cultivars in sunny locations whenever possible.Remove diseased foliage from the plant and clean up fallen debris on the ground.Prune or stake plants to improve air circulation.Make sure to disinfect your pruning tools (one part bleach to 4 parts water) after each cut.\n Milk sprays, made with 40% milk and 60% water, are an effective home remedy for use on a wide range of plants.For best results, spray plant leaves as a preventative measure every 10-14 days.</p>')
    elif(s=='Cherry_(including_sour)_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Cherry(including_sour) Leaf<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\n Just keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Corn_(maize)_Cercospora_leaf_spot Gray_leaf_spot'):
        return('<h1><font color="RED">Disease</font>: Corn(maize) cospora leaf-spot Grayleaf-spot <h1><h3><font color="GREEN">Remedy</h3></font><p>Crop rotation away from corn can reduce disease pressure, but multiple years may be necessary in no-till scenarios.Production practices such as tillage and crop rotation that reduce the amount corn residue on the surface will decrease the amount of primary inoculum.Planting hybrids with a high level of genetic resistance can help reduce the risk of yield loss due to gray leaf spot infection.Pioneer® brand hybrids and parent lines are improved through a screening process in areas with a high incidence of GLS and specialized “disease nurseries".Customers can see the effectiveness of hybrid resistance based off of a score (ranging from 1 to 9) that is assigned to Pioneer brand products.</p>')
    elif(s=='Corn_(maize)_Common_rust_'):
        return('<h1><font color="RED">Disease</font>: Corn(maize) Common Rust<h1><h3><font color="GREEN">Remedy</h3></font><p>Scout corn to detect common rust early.Monitor disease development, crop growth stage and weather forecast \n Disease is wind-borne and does not overwinter in U.S.; therefore, rotation and tillage are not effective.Modest control of rust on sweet corn can be achieved with applications of fungicides.</p>')
    elif(s=='Corn_(maize)_Northern_Leaf_Blight'):
        return('<h1><font color="RED">Disease</font>: Corn(maize) Northern Leaf Blight<h1><h3><font color="GREEN">Remedy</h3></font><p>Control of this disease is often focused on management and prevention. First, choose corn varieties or hybrids that are resistant or at least have moderate resistance to northern corn leaf blight.\n When you grow corn, make sure it does not stay wet for long periods of time. The fungus that causes this infection needs between six and 18 hours of leaf wetness to develop. Plant corn with enough space for airflow and water in the morning so leaves can dry throughout the day. \n Treating northern corn leaf blight involves using fungicides. For most home gardeners this step isn’t needed, but if you have a bad infection, you may want to try this chemical treatment. The infection usually begins around the time of silking, and this is when the fungicide should be applied.</p>')
    elif(s=='Corn_(maize)_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Corn(maize) <h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\n Just keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Grape_Black_rot'):
        return('<h1><font color="RED">Disease</font>: Grapes Black Rot<h1><h3><font color="GREEN">Remedy</h3></font><p>The disease attacks the leaves, stem, flowers and berries. All the new growth on the vineis prone to attack during the growing season.\n Occasionally, small elliptical darkcoloured canker lesions occur on the young stems and tendrils. Leaf, cane and tendril infection canoccur only when the tissue is young, but berries can be infected until almost fully-grown if an activefungicide residue is not present.\n The symptoms are in the form of irregularly shapedreddish brown spots on the leaves and a black scab on berries.The affected berries shrivel and become hard black mummies.</p>')
    elif(s=='Grape_Esca_(Black_Measles)'):
        return('<h1><font color="RED">Disease</font>: Grape Esca(Black Measels)<h1><h3><font color="GREEN">Remedy</h3></font><p>The disease attacks the leaves, stem, flowers and berries. All the new growth on the vineis prone to attack during the growing season.\n Occasionally, small elliptical darkcoloured canker lesions occur on the young stems and tendrils. Leaf, cane and tendril infection canoccur only when the tissue is young, but berries can be infected until almost fully-grown if an activefungicide residue is not present.\nThe symptoms are in the form of irregularly shapedreddish brown spots on the leaves and a black scab on berries.The affected berries shrivel and become hard black mummies.</p>')
    elif(s=='Grape_Leaf_blight_(Isariopsis_Leaf_Spot)'):
        return('<h1><font color="RED">Disease</font>: Grape Leaf-Blight(Isariopsis Leaf Spot)<h1><h3><font color="GREEN">Remedy</h3></font><p>The disease attacks the leaves, stem, flowers and berries. All the new growth on the vineis prone to attack during the growing season.\nOccasionally, small elliptical darkcoloured canker lesions occur on the young stems and tendrils. Leaf, cane and tendril infection canoccur only when the tissue is young, but berries can be infected until almost fully-grown if an activefungicide residue is not present.\nThe symptoms are in the form of irregularly shapedreddish brown spots on the leaves and a black scab on berries.The affected berries shrivel and become hard black mummies.</p>')
    elif(s=='Grape_healthy'):
        return('<h1><font color="RED">No Disease</font>:Healthy Grape <h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Orange_Haunglongbing_(Citrus_greening)'):
        return('<h1><font color="RED">Disease</font>: Orange Haunglongbing Citrus Greening<h1><h3><font color="GREEN">Remedy</h3></font><p>Once your citrus tree is infected, there is really nothing you can do about it. But steps are being taken to eradicate this devastating disease. Researchers are working hard to figure out how to control both the disease and the psyllid that transmits it. They are even working on developing citrus varieties that are resistant to HLB.</p>')
    elif(s=='Peach_Bacterial_spot'):
        return('<h1><font color="RED">Disease</font>: Peach Bacterial Spot<h1><h3><font color="GREEN">Remedy</h3></font><p>While there are no completely successful sprays for control of this disease, chemical spray with copper based bactericide and the antibiotic oxytetracycline have some effect used preventatively.\n Talk to your local extension office or nursery for information. Chemical control is doubtful, however, so the best long term control is to plant resistant cultivars</p>')
    elif(s=='Peach_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Peach<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Pepper,_bell_Bacterial_spot'):
        return('<h1><font color="RED">Disease</font>: Pepper Bacterial Spot<h1><h3><font color="GREEN">Remedy</h3></font><p>Treat seeds by soaking them for 2 minutes in a 10% chlorine bleach solution (1 part bleach; 9 parts water). Thoroughly rinse seeds and dry them before planting.\nMulch plants deeply with a thick organic material like newspaper covered with straw or grass clippings.Avoid overhead watering.\nRotate peppers to a different location if infections are severe and cover the soil with black plastic mulch or black landscape fabric prior to planting.</p>')
    elif(s=='Pepper,_bell_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Pepper<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Pepper,_bell_Bacterial_spot'):
        return('<h1><font color="RED">Disease</font>: Pepper Bell Bacterial Spot<h1><h3><font color="GREEN">Remedy</h3></font><p>The spores and mycelia of the pathogen survive in infested plant debris and soil, in infected tubers and in overwintering host crops and weeds.\nSpores are produced when temperatures are between 41-86 F. (5-30 C.) with alternating periods of wetness and dryness. These spores are then spread through wind, splashing rain and irrigation water. They gain entry via wounds caused by mechanical injury or insect feeding. Lesions begin to appear 2-3 days after the initial infection.</p>')
    elif(s=='Pepper,_bell_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Pepper<h1><h3><font color="GREEN">Remedy</h3></font><p>Infected tubers are the primary source of the pathogen P. infestans, including those in storage, volunteers, and seed potatoes. It is transmitted to newly emerging plants to produce airborne spores which then transmit the disease to nearby plants.\nUse only certified disease free seed and resistant cultivars where possible. Even when resistant cultivars are used, an application of fungicide may be warranted. Remove and destroy volunteers as well as any potatoes that have been culled.</p>')
    elif(s=='Potato_Early_blight'):
        return('<h1><font color="RED">Disease</font>: Potato Early Blight<h1><h3><font color="GREEN">Remedy</h3></font><p>Treatment of early blight includes prevention by planting potato varieties that are resistant to the disease; late maturing are more resistant than early maturing varieties</p>')
    elif(s=='Potato_Late_blight'):
        return('<h1><font color="RED">Disease</font>: Potato Late Blight<h1><h3><font color="GREEN">Remedy</h3></font><p>Prune or stake plants to improve air circulation and reduce fungal problems. Make sure to disinfect your pruning shears (one part bleach to 4 parts water) after each cut. Keep the soil under plants clean and free of garden debris.</p>')
    elif(s=='Potato_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Potatoes<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Raspberry_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Raspberry<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Soybean_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Soyabean<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Squash_Powdery_mildew'):
        return('<h1><font color="RED">Disease</font>: Squash Powdery Mildew<h1><h3><font color="GREEN">Remedy</h3></font><p>Along with the perfect storm of weather conditions, we no doubt aid and abet the disease. As mentioned above, the disease overwinters. Practicing a crop rotation will go a long way in preventing the spread of powdery mildew.\nDo not plant cucurbits in the same area for at least two years. We did not always practice crop rotation; I blame my other half.\nAdditional management techniques for treating powdery mildew in squash are to destroy any diseased plant debris, space plantings since a densely planted plot is more likely to be infected, and plant resistant varieties when possible. Also, keep the garden free of weeds. Powdery mildew control may also be need to be combined with a timely application of a fungicide.</p>')
    elif(s=="Strawberry_Leaf_scorch"):
        return('<h1><font color="RED">Disease</font>: Strawberry Leaf scorch<h1><h3><font color="GREEN">Remedy</h3></font><p>While leaf scorch on strawberry plants can be frustrating, there are some strategies which home gardeners may employ to help prevent its spread in the garden.The primary means of strawberry leaf scorch control should always be prevention.\nSince this fungal pathogen over winters on the fallen leaves of infect plants, proper garden sanitation is key. This includes the removal of infected garden debris from the strawberry patch, as well as the frequent establishment of new strawberry transplants. The creation of new plantings and strawberry patches is key to maintaining a consistent strawberry harvest, as older plants are more likely to show signs of severe infection.</p>')
    elif(s=='Strawberry_healthy'):
        return('<h1><font color="RED">No Disease</font>: Healthy Strawberry<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')
    elif(s=='Tomato_Bacterial_spot'):
        return('<h1><font color="RED">Disease</font>: Tomato Bacterial Spot<h1><h3><font color="GREEN">Remedy</h3></font><p>To develop an effectivediseasemanagement strategy, it is important to know how the bacterial pathogens differ from the fungal pathogens. Fungal pathogens multiply by spores, which are carried by wind or other means to new host plants, where they germinate and grow directly into the plant tissue. Bacterial pathogens are spread primarily by water and can travel long distances in wind-driven rain. Sprinkler irrigation systems also provide a means of disseminating bacteria. Once they arrive on the plant surface, bacteria must find a wound or natural opening to enter the plant and start the disease process. Bacteria multiply much more rapidly than fungi; under optimal conditions producing a new generation every 90 min.\nExperience has shown that if a bacterial disease outbreak can be delayed until after the main fruit set, the crop will be minimally affected. Once the plant has reached a full canopy, a low level of bacterial disease on the foliage can be tolerated. Fruit lesions, which have a maor impact on marketable yield, can only be initiated on young green fruit, so control measures used prior to fruiting are most beneficial .</p>')
    elif(s=='Tomato_Late_blight'):
        return('<h1><font color="RED">Disease</font>: Tomato Late Blights<h1><h3><font color="GREEN">Remedy</h3></font><p>Though no tomato varieties are completely immune to late blight, plant breeders are now developing varieties that are resistant to infection by the late blight fungus. So, when its time to decide which varieties to plant, keep an eye out for these. If you can, start your own plants from seed or buy transplants from a trusted local source. You might also want to plant some varieties that mature early so if late blight does strike, you may still get a harvest.\nFortunately, the fungus that causes late blight needs living tissue to survive over the winter, so it cant overwinter on tomato cages or supports. However, infected potatoes (the other plant that gets late blight) can carry the disease through the winter. Be sure to destroy any volunteer potato plants that come up. If you plant potatoes again, be sure to buy seed potatoes that are certified as disease-free.</p>')
    elif(s=='Tomato_Early_blight'):
        return('<h1><font color="RED">Disease</font>: Tomato Early Blights<h1><h3><font color="GREEN">Remedy</h3></font><p>Once you’ve seen the first signs of early blight affecting your plants, one of the best solutions is to apply a fungicide treatment.Since the disease is caused by a fungus, a fungicide is one of the most efficient solutions.\nYou need to closely watch for the first signs of early blight, which are the appearance of brown dark spots on the leaves at the bottom of the tomato plant.Remove the affected leaves (you can also remove any leaves that are very close or hanging on the soil) and throw them away or burn them once they dry out.\nDo not use those for your compost since you risk contaminating your next generation of tomatoes or potatoes when you use the compost.</p>')
    elif(s=='Tomato_Leaf_Mold'):
        return('<h1><font color="RED">Disease</font>: Tomato Leaf Mold<h1><h3><font color="GREEN">Remedy</h3></font><p>Remove and destroy all affected plant parts. For plants growing under cover, increase ventilation and, if possible, the space between plants. Copper-based fungicides can be used to control diseases on tomatoes.</p>')
    elif(s=='Tomato_Septoria_leaf_spot'):
        return('<h1><font color="RED">Disease</font>: Tomato Septoria leaf<h1><h3><font color="GREEN">Remedy</h3></font><p>Septoria is caused by a fungus, Septoria lycopersici, which overwinters in old tomato debris and on wild Solanaceous plants.\nThe fungus is spread by wind and rain, and flourishes in temperatures of 60 to 80 F. (16-27 C.).\nOld plant material needs to be cleaned up, and it’s best to plant tomatoes in a new location in the garden every year. One-year rotations of tomato plants have been shown to be effective in preventing the disease.</p>')
    elif(s=='Tomato_Spider_mites Two-spotted_spider_mite'):
        return('<h1><font color="RED">Disease</font>: Tomato Spider mites<h1><h3><font color="GREEN">Remedy</h3></font><p>Prune leaves, stems and other infested parts of plants well past any webbing and discard in trash (and not in compost piles). Don’t be hesitant to pull entire plants to prevent the mites spreading to its neighbors.\nCommercially available beneficial insects, such as ladybugs, lacewing and predatory mites are important natural enemies\nMix Pure Neem Oil with Coco-Wet and apply every 3-5 days to kill pest eggs indoors and interrupt the reproductive cycle. Make sure to spray all plant parts, including the undersides of leaves.</p>')
    elif(s=='Tomato_Target_Spot'):
        return('<h1><font color="RED">Disease</font>: Tomato Target_Spot<h1><h3><font color="GREEN">Remedy</h3></font><p>Most chemical fungicides are sprayed or dusted onto plant surfaces to protect against infection; they don’t kill fungus already infecting a plant.\nApply fungicides as soon as plants are in the ground and reapply weekly during the growing season and after a rain. Fungicides such as chlorothalonil, maneb and mancozeb are effective against blights and leaf spots.\ntarget spot is controlled primarily by applications of protectant fungicides. It should be noted that tank-mix sprays of copper fungicides and maneb do not provide acceptable levels of target spot control</p>')
    elif(s=='Tomato_Tomato_Yellow_Leaf_Curl_Virus'):
        return('<h1><font color="RED">Disease</font>: Tomato Yellow Leaf Curl Virus<h1><h3><font color="GREEN">Remedy</h3></font><p>Use only virus-and whitefly-freetomato and pepper transplants.Transplants should be treated with Capture (bifenthrin) or Venom (dinotefuran) for whitefly adults and Oberon for eggs and nymphs.Imidacloprid or thiamethoxam should be used in transplant houses at least seven days before shipping.Transplants should be produced in areas well away from tomato and pepper production fields.\nUse a neonicotinoid insecticide, such as dinotefuran (Venom) imidacloprid (AdmirePro, Alias, Nuprid, Widow, and others) or thiamethoxam (Platinum), as a soil application or through the drip irrigation system at transplanting of tomatoes.\nSanitation is very important for preventing the migration of whitefly adults and the spread of TYLCV. Rogue tomato or pepper plants with early symptoms of TYLCV can be removed from fields by placing infected-looking plants in plastic bags immediately at the beginning season, especially during first three to four weeks.</p>')
    elif(s=='Tomato_Tomato_mosaic_virus'):
        return('<h1><font color="RED">Disease</font>: Tomato mosaic virus<h1><h3><font color="GREEN">Remedy</h3></font><p>Use certified disease-free seed or treat your own seed.Soak seeds in a 10% solution of trisodium phosphate (Na3PO4) for at least 15 minutes.Avoid planting in fields where tomato root debris is present, as the virus can survive long-term in roots.\nScout plants regularly. If plants displaying symptoms of ToMV or TMV are found, remove the entire plant (including roots), bag the plant, and send it to the University of Minnesota Plant Diagnostic Clinic for diagnosis.\nIf ToMV or TMV is confirmed, employ stringent sanitation procedures to reduce spread to other plants, fields, tunnels and greenhouses.')
    else:
        print(len(s))
        print('disease name : ')
        print(s)
        print(type(s))
        return('<h1><font color="RED">No Disease</font>: Healthy Tomatoes<h1><h3><font color="GREEN">Remedy</h3></font><p>Its great !Your plant is healthy and good to be processed and consume further.\nJust keep your consistency on how you manage your plants and rest let it be dependent on the Climate and Environment.</p>')

#convert image_file to 5 cluster file
def convert_image_edge_cluster(img_path):
    
    #edge and green filter
    
    img=cv2.imread(img_path)
    img_g=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    #blure an image
    blur = cv2.blur(img_g,(10,10))  
    cv2.imwrite(r'D:\All_projects\plantdiseasedetection\static\clustor\test1.jpg', blur,[cv2.IMWRITE_JPEG_QUALITY, 100])
    
    #edge detection using guassian
    bw=img.mean(axis=2)
    hx=np.array([
    [-0.5,0,0.5],
    [-1.5,0,1.5],
    [-10,0,10]
    ],dtype=np.float32) 
    hy=hx.T 
    #vertiacl edge
    Gx=convolve2d(bw,hx)
    #horizontal edge
    Gy=convolve2d(bw,hy)
    cv2.imwrite(r'D:\All_projects\plantdiseasedetection\static\clustor\test4.jpg', Gx,[cv2.IMWRITE_JPEG_QUALITY, 100])
    
    #main edge
    G=np.sqrt(Gx*Gx + Gy*Gy)
    cv2.imwrite(r'D:\All_projects\plantdiseasedetection\static\clustor\test5.jpg', G,[cv2.IMWRITE_JPEG_QUALITY, 100])
    
    #Canny Edge
    edges = cv2.Canny(img,100,150)
    cv2.imwrite(r'D:\All_projects\plantdiseasedetectionstatic\clustor\test3.jpg', edges,[cv2.IMWRITE_JPEG_QUALITY, 100])
    
    #green filter
    green_image = img.copy() # Make a copy
    green_image[:,:,0] = 0
    green_image[:,:,2] = 0
    cv2.imwrite(r'D:\All_projects\plantdiseasedetection\static\clustor\test2.jpg', green_image,[cv2.IMWRITE_JPEG_QUALITY, 100])
    
    #clustering of image
    
    original_image = cv2.imread(img_path)
    img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
    #converts the MxNx3 image into a Kx3 matrix where K=MxN and each row is now a vector in the 3-D space of RGB
    vectorized = img.reshape((-1,3))
    #We convert the unit8 values to float as it is a requirement of the k-means method of OpenCV
    vectorized = np.float32(vectorized)
    #criteria: It is the iteration termination criteria. When this criterion is satisfied, the algorithm iteration stops.
    #Actually, it should be a tuple of 3 parameters
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)    
    x_data=[]
    attempts=10
    K=2
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    x_data.append(result_image)
    img_c = Image. fromarray(x_data[0])
    img_c.save(r'D:\All_projects\plantdiseasedetection\static\clustor\test6.jpg')
    K=3
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    x_data.append(result_image)
    img_c = Image. fromarray(x_data[1])
    img_c.save(r'D:\All_projects\plantdiseasedetection\static\clustor\test7.jpg')
    K=4
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    x_data.append(result_image)
    img_c = Image. fromarray(x_data[2])
    img_c.save(r'D:\All_projects\plantdiseasedetection\static\clustor\test8.jpg')
    K=5
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    x_data.append(result_image)
    img_c = Image. fromarray(x_data[3])
    img_c.save(r'D:\All_projects\plantdiseasedetection\static\clustor\test9.jpg')
    K=6
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    x_data.append(result_image)
    img_c = Image. fromarray(x_data[4])
    img_c.save(r'D:\All_projects\plantdiseasedetection\static\clustor\test10.jpg')
   

    

app.config['SECRET_KEY'] = 'the random string' 
if(local):
    app.config['SQLALCHEMY_DATABASE_URI'] = param['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = param['prod_uri']
db = SQLAlchemy(app)
class Users(db.Model):
	mobileno = db.Column(db.String(12), nullable=False,primary_key=True)
	name = db.Column(db.String(80), nullable=False)
	password = db.Column(db.String(12), nullable=False)
	totalland = db.Column(db.Integer)
	location = db.Column(db.String(100))

@app.route('/')
def home():
    return render_template('index.html',log=log,username=username)
@app.route('/about')
def about():
    return render_template('about.html',log=log)
@app.route('/learn')
def learn():
    return render_template('learn.html',log=log)
@app.route('/login',methods=['GET','POST'])
def login():
	if 'mob' in session:
		return redirect('/')
	if request.method == 'POST':
		mob=request.form.get('mobile')
		passw=request.form.get('password')
		u = Users.query.filter_by(mobileno=mob,password=passw).all()
		if(len(u)==1):
			session['mob']=mob
			global log
			log=True
			global username
			username=True

			return redirect('/')
		else:
			return render_template('login.html',error="Wrong username or password",log=log)
	return render_template('login.html',error="",log=log)
@app.route('/register',methods=['GET','POST'])
def register():
	if 'mob' in session:
		return redirect('/')
	if request.method=='POST':
		name=request.form.get('name')
		mob=request.form.get('mobile')
		passw=request.form.get('password')
		tl=request.form.get('plot')
		add=request.form.get('address')
		u=Users(name=name,mobileno=mob,password=passw,totalland=tl,location=add)
		db.session.add(u)
		db.session.commit()
		return redirect('/login')
	return render_template('register.html',log=log)
@app.route('/logout')
def logout():
	session.pop('mob')
	global log 
	log=False
	global username
	username=False
	return redirect('/')

@app.route('/analyse',methods=['GET','POST'])
def analyse():
	if request.method=='POST':
		f = request.files['leaf']
		basepath = os.path.dirname(__file__)
		file_path = os.path.join(basepath, 'static/uploads', secure_filename(f.filename))
		f.save(file_path),convert_image_edge_cluster(file_path);
		# Make prediction
		preds = model_predict(file_path, model)
		d = preds.flatten()
		j = d.max()
		for index,item in enumerate(d):
			if item == j:
				class_name = str(li[index])
		res = str(class_name).strip()
		print(class_name)
		print(type(class_name))
		result=convert(class_name)
		print(result)
		print(str("there is my result:"+str(result)))
		lis=['test6.jpg','test7.jpg','test8.jpg','test9.jpg','test10.jpg']
		return render_template('result.html',res=Markup(result),image_url=f.filename,clustor=lis)
	return render_template('analyse.html')
    
app.run(debug=True)
