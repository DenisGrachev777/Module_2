#!/usr/bin/env python
# coding: utf-8

# In[248]:


# в самом начале нам необохимо импортировать все библиотеки, которые планиуем использовать для нашего анализа
import numpy as np
import pandas as pd
from IPython.core.display import display
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import math
from IPython.display import Javascript
from itertools import combinations
from scipy.stats import ttest_ind

sns.set(style="whitegrid")
pd.set_option('display.max_rows', 50) # показывать больше строк
pd.set_option('display.max_columns', 50) # показывать больше колонок


# In[249]:


import warnings; warnings.simplefilter('ignore')
sns.set()


# ## Подготовка и преобрзование данных

# In[314]:


display(stud.head(10))
stud.info() 


# В датасете у нас 17 стрковых столбцов и 13 чиловых столбцов. Так же есть пустые значения Nan

# In[251]:


#Добавим в начале функции чтобы автоматизировать наши действия и сократить код
def about_col(column):
    display(stud[column].value_counts())
    stud.loc[:, [column]].info()
    print ('Значений, упомянутых более 10 раз:', (stud[column].value_counts()>10).sum())
    print ("Уникальных значений:", stud[column].nunique())
    print ('Значений NAN в столбце:', stud[column].isnull().sum(axis = 0)
)


# In[252]:


#Название столбцов нас устраивает. Однако для краоты и удобства использования заменим загланые буквы в названиях на маленькие
stud.columns = map(str.lower, stud.columns)
stud.columns


# In[253]:


#Посмотрим общей выгрузкой на наши пропущенные значения
#Почти все пропущенные значения в большом колличестве существуют в нечисловых столбцах - а значит их можно спокойно заменять. Мы сделаемэ то ниже
stud.isnull().sum(axis = 0)
    


# Пробуем рассмореть первые столбцы, чтобы выявить рекоммендации для обработки остальных столбцов

# In[255]:


#1 столбец - school - абревиатура шкаолы где учаться дети. У нас всего два уникальных вида занчений и нет пропусков тут.
about_col('school')
#малое кол-во уникальных значений


# In[256]:


#2 столбец - sex - распределение по полу. Возможно тут будет дальше интересная корреляция из за разницы колличества
about_col('sex')
#произведем вывод графика с дополнениями параметров вывода для наглядности
fig = plt.figure()
axes = fig.add_axes([0, 0, 0.5, 1])
axes.hist(stud['sex'], bins = 2)
#женский пол у нас преобладает


# Обощим наши процессы и выберем первичное рассмортение всех столбцов вместе

# Учитывая относительное малое количество данных попробуем просмотреть все столбцы сразу и при этом приименим для всех столбчатые диаграммы. Посмотрим как у них будут распределяться признаки.
# 
# Так же для того, чтобы удобнее было читать значения преобразуем некоторые параметры в столбцах как номинативных так к в тех, где не отображаются количественные показатели

# In[259]:


#Поместим наше правило по улучшению удобства порчитки графиков в специальный отдельный словарь

rules = {
    "sex": {
        "title": "пол",
        "values": {
          "F": 'девушки',
          "M": "юноши"   
        }
    },
    "address": {
        "title": "адрес",
        "values": {
          "U": "город",
          "R": "за городом"   
        }
    },
    "famsize": {
        "title": "размер семьи",
        "values": {
          "LE3": "<= 3",
          "GT3": "> 3"
        }
    },
    "pstatus": {
        "title": "родители вместе?",
        "values": {
          "T": "вместе",
          "A": "раздельно"
        }
    },
    "medu": {
        "title": "образование матери",
        "values": {
          "0.0": "0) нет",
          "1.0": "1) 4 кл",
          "2.0": "2) 5-9 кл",
          "3.0": "3) 11 кл / ср.сп.",
          "4.0": "4) высшее",
        }
    },
    "fedu": {
        "title": "образование отца",
        "values": {
          "0.0": "0) нет",
          "1.0": "1) 4 кл",
          "2.0": "2) 5-9 кл",
          "3.0": "3) 11 кл / ср.сп.",
          "4.0": "4) высшее",
        }
    },
    "fjob": {
        "title": "работа отца",
        "values": {
          "teacher": "учитель",
          "health": "здравохранение",
          "services": "гос.служба",
          "at_home": "не работает",
          "other": ".другое",
        }
    },
    "mjob": {
        "title": "работа матери",
        "values": {
          "teacher": "учитель",
          "health": "здравохранение",
          "services": "гос.служба",
          "at_home": "не работает",
          "other": ".другое",
        }
    },
    "reason": {
        "title": "причина выбора школы",
        "values": {
          "home": "близость к дому",
          "reputation": "репутация",
          "course": "об. программа",
          "other": ".другое",
        }
    },
    "guardian": {
        "title": "опекун",
        "values": {
          "mother": "мать",
          "father": "отец",
          "other": "другое",
        }
    },
    "traveltime": {
        "title": "время до школы (мин)",
        "values": {
          "1.0": "1) < 15",
          "2.0": "2) 15-30",
          "3.0": "3) 30-60",
          "4.0": "4) > 60",
        }
    },
    "studytime": {
        "title": "время учебы вне школы (ч)",
        "values": {
          "1.0": "1) < 2",
          "2.0": "2) 2-5",
          "3.0": "3) 5-10",
          "4.0": "4) > 10",
        }
    },
    "studytime, granular": {
        "title": "время учебы вне школы (ч) g",
        "values": {
          "1.0": "1) < 2",
          "2.0": "2) 2-5",
          "3.0": "3) 5-10",
          "4.0": "4) > 10",
        }
    },
    "failures": {
        "title": "внеучебные неудачи",
        "values": {
          "1.0": "1",
          "2.0": "2",
          "3.0": "3",
          "0.0": "другое",}
    },
    "schoolsup": {
        "title": "доп. обр. поддержка",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "famsup": {
        "title": "семейная обр. поддержка",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "paid": {
        "title": "доп. платная математика",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "activities": {
        "title": "доп. внеучебные занятия",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "nursery": {
        "title": "дет. сад",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "higher": {
        "title": "вышку хочет?",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "internet": {
        "title": "интернет есть?",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "romantic": {
        "title": "отношения",
        "values": {
          "yes": "да",
          "no": "нет"
        }
    },
    "famrel": {
        "title": "семейные отношения",
        "values": {
          "1.0": "1) ужасные",
          "2.0": "2) плохие",
          "3.0": "3) норм",
          "4.0": "4) хорошие",
          "5.0": "5) прекрасные",
        }
    },
    "freetime": {
        "title": "свобода",
        "values": {
          "1.0": "1) оч мало",
          "2.0": "2) мало",
          "3.0": "3) норм",
          "4.0": "4) много",
          "5.0": "5) оч много",
        }
    },
    "goout": {
        "title": "время с друзьями",
        "values": {
          "1.0": "1) оч мало",
          "2.0": "2) мало",
          "3.0": "3) норм",
          "4.0": "4) много",
          "5.0": "5) оч много",
        }
    },
    "health": {
        "title": "здоровье",
        "values": {
          "1.0": "1) ужасное",
          "2.0": "2) плохое",
          "3.0": "3) норм",
          "4.0": "4) хорошее",
          "5.0": "5) прекрасное",
        }
    }
}


# Теперь применим правило для неокторых столбцов так же напишем функцию для изменения пустых значений. Мы будем заменять значения в нечисловых столбцах на '-', так как результат тут не зависит от цифр, а небольшое колличество пропусков в числовых столбцах заменим на -1, это будет наше пустое значение

# In[428]:


#применим наш созданный ранее словарь
def change_column_type(name, type):
    if name in list(rules.keys()):
        if type in ['float64', 'int64']:
            stud[name] = stud[name].astype(str)

def smart_value(name, value):
    if name in list(rules.keys()):
        if value in list(rules[name]['values'].keys()):
            return rules[name]['values'][value]
    return value

#переберем и заменим наши неизвестные значения
def empty_value(type, value):
    if type == 'object':
        return '-'
    elif type in ['float64', 'int64']:
        return -1

def change_value(name, type, value):
    if value != value or value == 'nan':
        return empty_value(type, value)
    else:
        return smart_value(name, value)


for column_name in stud:
    change_column_type(column_name, stud[column_name].dtype)
    stud[column_name] = stud[column_name].map(
      lambda value: change_value(column_name, stud[column_name].dtype, value)
  )


# Мы провели преобразование столбцов и данных и теперь можем посмотреть на наши данные ввиде диаграмм. Как писали выше, что нам вполне подходит тут тип диаграмм ввиде столбцов

# In[429]:


#Напишем функйию для вывод графиков # #Подготовим снначала место для вывода значений
def explore(stud, columns, inrow = 4, percent = True):
  fig, axes = plt.subplots(
    math.ceil(len(columns)/inrow), 
    inrow, 
    figsize=(24, math.ceil(len(columns)/inrow) * 4)
  )

  for column in columns:
    if len(columns) > len(columns) / inrow:
      ax = axes[
        int(math.floor(columns.index(column) + 1) / inrow - 0.00000001),
        int(columns.index(column) % inrow)
      ]
    else:
      ax = axes[int(math.floor(columns.index(column) + 1) / inrow - 0.00000001)]

    stud.sort_values(by = [column], inplace = True)
    countplot = sns.countplot(data = stud, x = column, ax = ax)
  
    countplot.set(
      ylim=(0, len(stud)),
      xlabel = rules[column]['title'] if column in list(rules.keys()) else column, 
      ylabel=''
    )
  
    for p in countplot.patches:
        x=p.get_bbox().get_points()[:,0]
        y=p.get_bbox().get_points()[1,1]
        countplot.annotate(
            f'{int(y)} {f"({int(round(100*y/len(stud), 0))}%)" if percent else ""}', 
            (x.mean(), y), 
            ha ='center', 
            va='bottom'
        )
  
  fig.tight_layout() 
  fig.show()

explore(stud, ['school', 'sex', 'address', 'famsize', 'pstatus', 
  'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
  'higher', 'internet', 'romantic'], 6)

explore(stud, ['age', 'medu', 'fedu', 'mjob', 'fjob', 'reason', 
  'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 
  'goout', 'health'], 2)

explore(stud, ['absences', 'score'], 1, percent = False)


# Благодаря выбранному типу диаграмм мы видим особенности и отличия различных признаков друг от друга. Давайте рассмотрим какие признаки и какие колличественные показатели у нас преобладают в наших
# таблицах сравнения:
#  - Большенсство учаться в школе  Gp
#  - Девушек больше чем парней с рассматриваемой выборке
#  - Живут почте все городе
#  - В семье больше 3х человек и при этом родители не в разводе
#  - Дополнительное образование почти никто не использует, но основное образование спонсирует семья
#  - Но большенство не занимаются дополнительно по математике, но занимаются другими занятиями вне учебное время
#  - Почти все все были в детсаду и почти все устремлены поступить в высшее учебное заведение
#  - У большенства есть интернет и нет отношений 
#  - Основной возраст 16 лет
#  - В большенстве рассмотренных семей у матерей есть высшее образование, но у отцов только 5-9 класс
#  - Выбирают школу потому что нравиться образовательная программа и при этом идти до школы меньше 15 минут
#  - Большенство занимаются учебой вне школы от 2-до 5 часов,
#  - Почти у всех хорошие и даже прекрасные отношения дома и чувствую умеренное кол-во личной свободы, и имеют прекрасное здоровье
#  
#  
#  Дополнительно по графикам можно увидеть предполагаемые выбросы в столбцах
#  - семейные отношения
#  - возраст
#  - образование матери
#  - образование отца 
# 
# Не понятные значения получили в стобцах 
# - образование отца - правый выбро - надо будет проверить что это
# - семеный отношения - значение - 1 - не понятно 
# -
# 
# 

# In[430]:


# Сделаем преобразования в столбце семейные отношения

stud['famrel'][stud['famrel'] == '-1.0'] = rules['famrel']['values']['1.0']


# In[431]:


# Cделаем преобразования в столбце образование отца 

stud['fedu'][stud['fedu'] == '40.0'] = rules['fedu']['values']['4.0']


# ## Выбросы

# In[432]:


#Будем искать выбросы в числовых столбцах. Берем интерквартильное расстояние и применяем формулу 


# In[433]:


def show_outliners():
  for column_name in stud:
    column = stud[column_name]
    if column.dtype in ['int64', 'float64']:
      IQR = column.quantile(0.75) - column.quantile(0.25)
      outliners = {
        "left": column.quantile(0.25) - 1.5 * IQR, 
        "right": column.quantile(0.75) + 1.5 * IQR
      }
      print(column_name, outliners)

    
show_outliners()


#  Итого у нас выбросы в трех числовых столбцах, которые нам предлагает функция увидеть:
#  
#  - в возрасте это значение 21
#  - в пропусках выбросы слева и справа 
#  - и показывает выбросы в столбцах оценки
# 
# Рассмотрим эти столбцы дополнительно

# In[434]:


stud.age.value_counts()


# In[435]:


#будем считать что 20 21 и 22 это все очень маленькие значения относительноосновных
#проработаем эти данные и примем их все за значение 20
stud['age'][stud['age'] >= 20] = 20


# In[436]:


stud.score.value_counts()
#выбросов нет


# In[438]:


print(stud.absences.value_counts())
#преобразуем наши выбромым 212 и 385. А все что выше 20 соединим в одно
stud["absences"][stud["absences"] > 200] = -2
stud["absences"][stud["absences"] > 20] = 20


# ## Корреляция

# In[439]:


#Рассматриваем графики с числовыми значенияи
sns.pairplot(stud, kind = 'reg')


# Сразу бросаются в глаза корреляция 
# - по возрасту
# - по пропускам
# 
# Подробнее получается вот что:
# Чем старше человек тем больше пропукает и хуже сдает экзамен
# Чем больше человек пропускает- тем лучше он сдает экзамен
# 
# 
# Так же слева снизу на графике оценки есть показатели 0 - что можно смело удалять и не учитывать в нашей выборке

# In[440]:


stud1 = stud[stud['score'] > 0]
sns.pairplot(stud1, kind = 'reg')


# Теперь больше видна зависимоть возраста пропусков и итогой оценки

# In[441]:


#Дополним нашу провероку метобом hue
sns.pairplot(stud1, hue = 'age')


# In[442]:


#Проверим наши гипотезы наглядно еще одним способом, однако дополнительно что-то мы не наблюдаем
fig = plt.figure()
axes = fig.add_axes([0, 0, 1, 1])
axes.scatter(x = stud1['age'], y = stud1['score'])


# ## Анализ номинативных переменных

# In[443]:


def countplots(stud1):
  fig, axes = plt.subplots(9, 3, figsize = (22, 60))

  count = 0

  for column_name in stud:
    if stud1[column_name].dtype == 'object':

      ax = axes[count // 3, count % 3]

      df = stud1.sort_values(by = [column_name])

      sns.boxplot(data = stud1, x = column_name, y = "score", ax = ax)

      count += 1

  fig.tight_layout() 
  fig.show()


countplots(stud1)


# Итого:
# 
# Опишем признаки и значения тех, у кого лучшие результаты
# - из школы GP
# - Парни
# - С образованием матерей гос служба 
# - С образованием отцов учителя
# - Выюирающие школу по репутации и близости к дому
# - Много учатся
# - Посещали детский сад
# - Мало гуляют и без отношений
# - С плохим здоровьем
# - и д.р.
# 

# # Тест Стюдента

# In[444]:


for col in stud1.columns:
    
    combs = list(combinations(stud1[col].unique(), 2))
    
    for a, b in combs:
        a_values = stud1.loc[stud[col] == a, 'score']
        b_values = stud1.loc[stud[col] == b, 'score']        
        pval = ttest_ind(a_values, b_values).pvalue
        if pval <= (0.05 / len(combs)):
            print(f"Найдены статистически значимые различия для колонки {col}")
            break


# Давайте оставим тогда для нашей таблицы только значимые статистически данные 

# In[445]:


finall_stud = stud1.loc[:, ['address', 'age', 'medu', 'fedu', 'mjob', 'fjob', 'studytime', 'failures', 'schoolsup', 'goout', 'absences', 'score']]
finall_stud


# В результате EDA анализа мы провели анализ данных и получили следующие выводы:
#     
# В данных достаточно мало уникальных значений по каждому столбцу поэтому анализировать зависимости получается с большей точностью
# 
# Выбросмы были найдены только в столбцах с 'age' и 'score', что позволяет сделать вывод, что данные достаточно чистые
# 
# Положительныя корреляция параметров 'age' 'score' и 'absences' может говорить о том, что с годами оценки ухудшаются как и с пропусками занятий. 
# 
# Самые важные параметры, которые предлагается использовать в дальнейшем для построения модели, это 'address', 'age', 'medu', 'fedu', 'mjob', 'fjob', 'studytime', 'failures', 'schoolsup', 'goout', 'absences', 'score'

# In[ ]:





# In[ ]:




