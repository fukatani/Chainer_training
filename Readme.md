
Introduction
==============================
Chainer及びRaspberry Pi勉強用のコード集です。
ほぼ自分と知人用のため、ドキュメントは分かる人用です。


Software Requirements
==============================
* Python 2.7
* numpy, scipy, matplotlib, h5py
→特にWindowsの方はexe形式のインストーラでこれらをインストールしてからchainerをインストールしてください。インストーラはググって探しましょう。
* Chainer
→できれば1.5.X。ただし1.5系はインストールトラブルが多いので、1.2.0でもいいです。
最新版を入れるときは、
```
$ pip install chainer
```

1.2.0を入れるときは、
```
$ pip install chainer=1.2.0
```

をします。

* git
必要です。
Windowsの方はSourceTreeをインストールして、一緒にgitも入れちゃってください。

Installation
==============================

```
git clone https://github.com/fukatani/Chainer_training.git
```

コミットしたい方はフォークして、プルリクエストしてください。

Mychainを継承しているオブジェクトを実行すると動きます。
==============================


License
==============================

Apache License 2.0
(http://www.apache.org/licenses/LICENSE-2.0)


Copyright
==============================

Copyright (C) 2015, Ryosuke Fukatani

