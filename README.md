# EECE5644_IntroMLPR_LectureCode

Code to complement the _"EECE 5644: Introduction to Machine Learning and Pattern Recognition"_ course taught at Northeastern University over the Summer 2022 semesters. You can find some of the lectures publicly available at [this webpage](https://markzolotas.com/introduction-to-machine-learning-pattern-recognition/).

Please contact the lecturer [Mark Zolotas](https://markzolotas.com/contact/) about any questions concerning the repository.

## Directory Layout

- `data` to hold any datasets used by models in the codebase, which is purposefully not Git tracked to avoid a large repository.
- `docs` have question sheets for any course assignments, adapted from the previous professor's (Deniz Erdogmus) original homeworks.
- `matlab` for Matlab code that replicates _some_ of the Python implementations.
- `notebooks` consists of Jupyter Notebook files that walk through examples with both mathematical formulation and programming implementation.
- `scripts` are example Python programs that test the functionality provided by the `modules` library. These examples are less complete than the Notebooks.

## Getting Started

You can "clone" this Git repository into your workspace, by running the following command:
```
git clone https://github.com/mazrk7/EECE5644_IntroMLPR_LectureCode.git
```
If you wish to "pull" any updates I make to this repository over time, run:
```
git pull origin main
```
Look through this [Git Basics](https://git-scm.com/book/en/v2/Git-Basics-Getting-a-Git-Repository) if you would like further guidance.

I make use of Jupyter Notebooks for many of the in-class demonstrations. This is basically a Python web interface that allows me to combine written formulas/notes with code and run it in a sequential fashion. My environment is set up using [Anaconda](https://www.anaconda.com/), which I highly recommend for hosting applications like Jupyter within your Python [environment](https://docs.python.org/3/library/venv.html#:~:text=A%20virtual%20environment%20is%20a,part%20of%20your%20operating%20system.).

Ofcourse, you can also look through the Notebooks without running the code. Or you can directly import the code aspects into whatever Python IDE is of your preference. I usually recommend these two:
- [Visual Studio Code](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)

Both are great so take your pick!

Lastly, my Python environment can be installed directly using `pip` as follows:
```
pip install -r requirements.txt
```
Or by creating a Conda environment from my provided `yaml` file:
```
conda env create -f environment.yaml
```
Please note that these environments were generated for the **Python 3.8.13** release and may require dependency consolidation if you use another Python version.

Setting up interactive plots in jupyter
`ipympl` enables the interactive features of matplotlib in the Jupyter notebook. Instructions are in its [official repo](https://github.com/matplotlib/ipympl).

## Further Reading

Many of the code and examples presented in this repository take inspiration from [Kevin Murphy's "Probabilistic Machine Learning: An Introduction", 2022 book](https://probml.github.io/pml-book/book1.html). I highly recommend this book for further reading.