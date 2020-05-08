NeuralNetworks
=====================
**NeuralNetworks** - это фрэймворк для работы с нейронными сетями на языке C++.

С помощью него можно работать с такими архитектурами нейронных сетей как: 
1. [Перцептрон](https://ru.wikipedia.org/wiki/%D0%9F%D0%B5%D1%80%D1%86%D0%B5%D0%BF%D1%82%D1%80%D0%BE%D0%BD)
2. [Глубокии сети прямого распространения](https://neuralnet.info/chapter/%D0%BE%D1%81%D0%BD%D0%BE%D0%B2%D1%8B-%D0%B8%D0%BD%D1%81/)
3. [Свёрточные нейронные сети](https://ru.wikipedia.org/wiki/%D0%A1%D0%B2%D1%91%D1%80%D1%82%D0%BE%D1%87%D0%BD%D0%B0%D1%8F_%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C)
4. [Самоорганизующаяся карта Кохонена](https://ru.wikipedia.org/wiki/%D0%A1%D0%B0%D0%BC%D0%BE%D0%BE%D1%80%D0%B3%D0%B0%D0%BD%D0%B8%D0%B7%D1%83%D1%8E%D1%89%D0%B0%D1%8F%D1%81%D1%8F_%D0%BA%D0%B0%D1%80%D1%82%D0%B0_%D0%9A%D0%BE%D1%85%D0%BE%D0%BD%D0%B5%D0%BD%D0%B0)
5. [Сети Хопфилда](https://ru.wikipedia.org/wiki/%D0%9D%D0%B5%D0%B9%D1%80%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D1%81%D0%B5%D1%82%D1%8C_%D0%A5%D0%BE%D0%BF%D1%84%D0%B8%D0%BB%D0%B4%D0%B0)

Инструкция по установке:
----------------------
1. Скачать репозиторий
2. Установить [OpenCV](https://github.com/opencv/opencv) и [Boost](https://github.com/boostorg/boost)
3. Прописать пути к библиотекам что бы Cmake нашел их
4. Скачать рахив [mnist_png.zip](https://www.kaggle.com/jidhumohan/mnist-png/data#)
5. Распаковать его в корень репозитория
6. В корне репозитория выполнить mkdir build && cd ./build && cmake build ..
