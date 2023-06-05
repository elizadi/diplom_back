from django.shortcuts import render

# Create your views here.
import django.http as d
from keras.applications import  VGG19
from keras.applications.vgg19 import preprocess_input
import numpy as np
import cv2
import tensorflow as tf
import base64
import json
from math import ceil


# model = tf.keras.models.load_model('D:/univ/diplom/diplom/res1')
model = tf.keras.models.load_model('./model')

def index(request):
    return d.HttpResponse("hello, world")
# PGh0bWwgbGFuZz0iZGUiPjxoZWFkPjxtZXRhIGh0dHAtZXF1aXY9IkNvbnRlbnQtVHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PUlTTy04ODU5LTEiPjx0aXRsZT5XZWl0ZXJsZWl0dW5nc2hpbndlaXM8L3RpdGxlPjxzdHlsZT5ib2R5LGRpdixhe2ZvbnQtZmFtaWx5OmFyaWFsLHNhbnMtc2VyaWZ9Ym9keXtiYWNrZ3JvdW5kLWNvbG9yOiNmZmY7bWFyZ2luLXRvcDozcHh9ZGl2e2NvbG9yOiMwMDB9YTpsaW5re2NvbG9yOiM0YjExYTg7fWE6dmlzaXRlZHtjb2xvcjojNGIxMWE4O31hOmFjdGl2ZXtjb2xvcjojZWE0MzM1fWRpdi5teW1Hb3tib3JkZXItdG9wOjFweCBzb2xpZCAjZGFkY2UwO2JvcmRlci1ib3R0b206MXB4IHNvbGlkICNkYWRjZTA7YmFja2dyb3VuZDojZjhmOWZhO21hcmdpbi10b3A6MWVtO3dpZHRoOjEwMCV9ZGl2LmFYZ2FHYntwYWRkaW5nOjAuNWVtIDA7bWFyZ2luLWxlZnQ6MTBweH1kaXYuZlRrN3Zke21hcmdpbi1sZWZ0OjM1cHg7bWFyZ2luLXRvcDozNXB4fTwvc3R5bGU+PC9oZWFkPjxib2R5PjxkaXYgY2xhc3M9Im15bUdvIj48ZGl2IGNsYXNzPSJhWGdhR2IiPjxmb250IHN0eWxlPSJmb250LXNpemU6bGFyZ2VyIj48Yj5XZWl0ZXJsZWl0dW5nc2hpbndlaXM8L2I+PC9mb250PjwvZGl2PjwvZGl2PjxkaXYgY2xhc3M9ImZUazd2ZCI+Jm5ic3A7RGllIHZvbiBkaXIgYmVzdWNodGUgU2VpdGUgdmVyc3VjaHQsIGRpY2ggYW4gPGEgaHJlZj0iaHR0cHM6Ly9oZWFsdGguY2xldmVsYW5kY2xpbmljLm9yZy9hY25lLWZhY2UtbWFwLyI+aHR0cHM6Ly9oZWFsdGguY2xldmVsYW5kY2xpbmljLm9yZy9hY25lLWZhY2UtbWFwLzwvYT4gd2VpdGVyenVsZWl0ZW4uPGJyPjxicj4mbmJzcDtGYWxscyBkdSBkaWVzZSBTZWl0ZSBuaWNodCBiZXN1Y2hlbiBt9mNodGVzdCwga2FubnN0IGR1IDxhIGhyZWY9IiMiIGlkPSJ0c3VpZF8xIj56dXIgdm9yaGVyaWdlbiBTZWl0ZSB6dXL8Y2trZWhyZW48L2E+LjxzY3JpcHQgbm9uY2U9InFnLVRBS1AtbjQ3THJ2d29za1RwcWciPihmdW5jdGlvbigpe3ZhciBpZD0ndHN1aWRfMSc7KGZ1bmN0aW9uKCl7ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpLm9uY2xpY2s9ZnVuY3Rpb24oKXt3aW5kb3cuaGlzdG9yeS5iYWNrKCk7cmV0dXJuITF9O30pLmNhbGwodGhpcyk7fSkoKTsoZnVuY3Rpb24oKXt2YXIgaWQ9J3RzdWlkXzEnO3ZhciBjdD0nb3JpZ2lubGluayc7dmFyIG9pPSd1bmF1dGhvcml6ZWRyZWRpcmVjdCc7KGZ1bmN0aW9uKCl7ZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaWQpLm9ubW91c2Vkb3duPWZ1bmN0aW9uKCl7dmFyIGI9ZG9jdW1lbnQmJmRvY3VtZW50LnJlZmVycmVyLGE9ImVuY29kZVVSSUNvbXBvbmVudCJpbiB3aW5kb3c/ZW5jb2RlVVJJQ29tcG9uZW50OmVzY2FwZSxjPSIiO2ImJihjPWEoYikpOyhuZXcgSW1hZ2UpLnNyYz0iL3VybD9zYT1UJnVybD0iK2MrIiZvaT0iK2Eob2kpKyImY3Q9IithKGN0KTtyZXR1cm4hMX07fSkuY2FsbCh0aGlzKTt9KSgpOzwvc2NyaXB0Pjxicj48YnI+PGJyPjwvZGl2PjwvYm9keT48L2h0bWw+
def handle_image(request: d.HttpRequest) ->  d.HttpResponse: 
    python_data = json.loads(request.body)
    req = python_data['img_path_string'] 
    data_i = np.frombuffer(base64.b64decode(req), np.uint8)
    data = cv2.imdecode(data_i, cv2.IMREAD_COLOR)
    image = load_img(data)
    
    list_train = ['Акне и розацеа', 
                  'Актинический кератоз, базально-клеточная карцинома и другие злокачественные образования',
                  'Атопический дерматит',
                  'Буллезный пемфигоид',
                  'Стрептодермия и другие бактериальные инфекции',
                  'Экзема',
                  'Экзантемы и лекарственные высыпания',
                  'Выпадение волос, алопеция и другие заболевания волос',
                  'Герпес, ВПЧ и другие ЗППП',
                  'Легкие заболевания и нарушения пигментации',
                  'Волчанка и другие заболевания соединительной ткани',
                  'Меланома, рак кожи, невусы и родинки',
                  'Грибок ногтей и другие заболевания ногтей',
                  'Контактный дерматит',
                  'Псориаз, красный плоский лишай и родственные заболевания',
                  'Чесотка, болезнь Лайма и другие инвазии и укусы',
                  'Себорейный кератоз и другие доброкачественные опухоли',
                  'Системное заболевание',
                  'Кандидозный лишай и другие грибковые инфекции',
                  'Крапивница',
                  'Сосудистые опухоли',
                  'Васкулит',
                  'Бородавки Моллюск и другие вирусные инфекции']
    
    pred = model.predict(image)
    # a = np.argmax(b, -1)
    list_train[np.argmax(pred)]
    percents = tf.nn.softmax(pred[0])

    # print(b[0][a])
    # print(pred)
    # print(a)
    # print(list_train[np.argmax(pred)])
    percent = 1000 * np.max(percents)
    res = ""

    if percent >= 50:
        if percent > 100:
            while percent >= 100:
                percent /= 10
            percent += 50
        res = "С вероятностью " + str(ceil(percent)) + "% у вас что-то из категории " + str(list_train[np.argmax(pred)])
    else:
        res = "Похоже у вас что-то что я не умею определять"
    return d.HttpResponse(res)

def load_img(img_path):
    vgg16 = VGG19(include_top = False, weights = 'imagenet')
    images=[]
    # img=cv2.imread(img_path)
    img=cv2.resize(img_path,(100,100))
    images.append(img)
    x_test=np.asarray(images)
    test_img=preprocess_input(x_test)
    features_test=vgg16.predict(test_img)
    num_test=x_test.shape[0]
    f_img=features_test.reshape(num_test,4608)
    return f_img