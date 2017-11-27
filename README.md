# MACS 40000: Economic Policy Analysis with Overlapping Generations Models (Autumn 2017) #

|  | [Dr. Richard Evans](https://sites.google.com/site/rickecon/) |
|--------------|--------------------------------------------------------------|
| Email | rwevans@uchicago.edu |
| Office | 250 Saieh Hall |
| Office Hours | W 10:00am-noon |
| GitHub | [rickecon](https://github.com/rickecon) |

* **Meeting day/time**: M,W 1:30-2:50pm, Swift Hall, Room 208
* Office hours also available by appointment

## Prerequisites ##

Advanced undergraduate or first-year graduate microeconomic theory, linear algebra, multivariable calculus, recommended coding experience.


## Texts ##

* Evans, Richard W. and Jason DeBacker, *Overlapping Generations Models for Policy Analysis: Theory and Computation*, unpublished draft (2017).


## Course description ##

This course will study economic policy questions ideally addressed by the overlapping generations (OG) dynamic general equilibrium framework. OG models represent a rich class of macroeconomic general equilibrium model that is extremely useful for answering questions in which inequality, demographics, and individual heterogeneity are important. OG models are used extensively by the Joint Committee on Taxation, Congressional Budget Office, Department of the Treasury, and Federal Reserve System for policy analysis in the United States as well as in other developed nations.

This course will train students how to set up and solve OG models. The standard nonlinear global solution method for these models--time path iteration--is a fixed point method that is similar to but significantly different from value function iteration and policy function iteration. This course will take students through progressively richer versions of the model, which will include endogenous labor supply, nontrivial demographics, bequests, stochastic income, multiple industries, non-balanced government budget constraint, and household tax structure.

We will be focusing on computational strategies, modularity of code, sensitivity and robustness to assumptions, and policy questions that can be answered with this framework. Students can use whatever programming language they want, but I highly recommend you use Python 3.x ([Anaconda distribution](https://www.anaconda.com/download/)). I will be most helpful with code debugging and suggestions in Python. We will also study results and uses from recent papers listed in the "References" section below. The dates on which we will be covering those references are listed in the "Daily Course Outline" section below.


## Course Objectives and Learning Outcomes ##

* You will know how to augment the standard overlapping generations model framework to answer policy questions.
* You will know the computational methods to solve for the steady-state and transition path equilibria of the model.
* You will learn computational tools such as:
	* Constrained minimization
	* Multi-equation root finder with occasionally binding constraints
	* Interpolation
	* Curve fitting
	* Parametric and nonparametric estimation
* You will learn coding and collaboration techniques such as:
	* Best practices for Python coding ([PEP 8](https://www.python.org/dev/peps/pep-0008/))
	* Writing modular code with functions and objects
	* Creating clear docstrings for functions
	* Collaboration tools for writing code using [Git](https://git-scm.com/) and [GitHub.com](https://github.com/).


## Grades ##

Grades will be based on the four categories listed below with the corresponding weights.

Assignment   | Points | Percent |
-------------|--------|------|
Homework     |   100  |  50% |
Midterm      |    50  |  25% |
Final Exam   |    50  |  25% |
**Total points** | **200** | **100%** |

* **Homework:** I will assign 8 problem sets throughout the term, and I will drop your one lowest problem set score.
	* You must write and submit your own computer code, although I encourage you to collaborate with your fellow students. I **DO NOT** want to see a bunch of copies of identical code. I **DO** want to see each of you learning how to code these problems so that you could do it on your own.
	* Problem set solutions, both written and code portions, will be turned in via a pull request from your private [GitHub.com](https://git-scm.com/) repository which is a fork of the class master repository on my account. (You will need to set up a GitHub account if you do not already have one.)
	* Problem sets will be due on the day listed in the Daily Course Outline section of this syllabus (see below) unless otherwise specified. Late homework will not be graded.
* **Midterm:** The midterm will be given on Wednesday, October 25, during class and will cover the material up to that point in the course.
* **Final Exam:** The final exam will be comprehensive and will be given on Wednesday, Dec. 6, from 1:30 to 3:30p.m. in our classroom (Swift 208).


## Daily Course Schedule ##

| Date | Day | Topic | Readings | Homework |
|------|-----|-------|----------|----------|
| Sep. 25 | M | Python, Git, OG Models        | Ch. 1, [tutorials](https://github.com/rickecon/OGcourse_F17/tree/master/Tutorials)   |      |
|         |   |                               | Weil (2008) |      |
|         |   |                               | N&S (2007)  |      |
| Sep. 27 | W | 3-period-lived model: theory  | Ch. 5       | [PS 1](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS1/PS1.pdf) |
| Oct.  2 | M | 3-period-lived model: theory  | Ch. 5       |      |
| Oct.  4 | W | 3-period-lived model: computation | Ch. 5   |      |
| Oct. 9 | M | *S*-period-lived model        | Ch. 6       |      |
| Oct. 11 | W | Endogenous labor supply       | Ch. 7       | [PS 2](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS2/PS2.pdf) |
|         |    |                               | P (2016)    |      |
|         |    |                               | E&P (2016)  |      |
| Oct. 16 | M | Endogenous labor supply       | Ch. 7       | [PS 3](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS3/PS3.pdf) |
| Oct. 18 | W | Endogenous labor supply       | Ch. 7       |      |
| Oct. 23 | M | Endogenous labor supply       | Ch. 7       |      |
| **Oct. 25** | **W** | **Midterm 1 (Chs. 1-7)** |          | [PS 4](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS4/PS4.pdf) |
| Oct. 30 |  M | Bequests: simple              | Ch. 11      |      |
|         |    |                               | D (2015)    |      |
|         |    |                               | DEMPRS (2016) |    |
| Nov. 1 | W | Bequests: simple                | Ch. 11 |      |
| Nov.  6 | M  | Bequests: calibrated from SCF |  Ch. 11 | [PS5](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS5/PS5.pdf)  |
|         |    |                               | N (2015)    |      |
| Nov. 8 | W | Population demographics       | Ch. 10 | [PS6](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS6/PS6.pdf) |
| Nov. 13 | M | Population demographics       | Ch. 10 |      |
| Nov. 15 | W | Population demographics |  Ch. 10 | [PS7](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS7/PS7.pdf) |
|         |    |                               | DEP (2016)  |      |
| Nov. 20 | M | Tax functions in OG model | Ch. 14 |   |
| Nov. 22 | W | Tax functions from microsimulation | Ch. 14 | [PS 8](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS8/PS8.pdf) |
| Nov. 27 | M | Tax functions from microsimulation | Ch. 14 |      |
| Nov. 29 | W | Multiple industry model | Ch. 17   | [PS9](https://github.com/rickecon/OGcourse_F17/blob/master/ProblemSets/PS9/PS9.pdf)  |
|         |    | *Exam preparation (reading) days, Nov. 30-Dec. 1* |  |   |
| **Dec. 6** | **W** | **Final Exam (comprehensive)** |      |      |
|         |     | **1:30-3:30p.m. in Swift 208** |           |      |


## References ##

* De Nardi, Mariacristina, "[Quantitative Models of Wealth Inequality: A Survey](http://users.nber.org/~denardim/research/NBERwp21106.pdf)," National Bureau of Economic Research, NBER Working Paper 21106 (April 2015).
* DeBacker, Jason, Richard W. Evans, Evan Magnusson, Kerk L. Phillips, Shanthi Ramnath, and Isaac Swift, "[The Distributional Effects of Redistributional Tax Policy](https://sites.google.com/site/rickecon/WealthTax.pdf)," under review at *Quantitative Economics* (August 2016).
* DeBacker, Jason, Richard W. Evans, and Kerk L. Phillips, "[Integrating Microsimulation Tax Functions into a DGE Macroeconomic Model: A Canonical Example](https://sites.google.com/site/rickecon/DEP_10pct.pdf)," mimeo (August 2016).
* Evans, Richard W. and Jason DeBacker, *Overlapping Generations Models for Policy Analysis: Theory and Computation*, unpublished draft (2017).
* Evans, Richard W. and Kerk L. Phillips, "[Advantages of an Ellipse when Modeling Leisure Utility](https://sites.google.com/site/rickecon/Elliptical.pdf)," *Computational Economics*, (forthcoming, 2016).
* Furman, Jason, "Can Tax Reform Get Us to 3 Percent Growth?" [[Slides](https://github.com/rickecon/OGcourse_F17/blob/master/Slides/furman20171103ppt.pdf)], Talk at New York Association for Business Economics (November 3, 2017).
* Nishiyama, Shinichi, "Fiscal Policy Effects in a Heterogeneous-agent OLG economy with an Aging Population," *Journal of Economic Dynamics & Control*, 61, pp. 114-132 (December 2015).
* Nishiyama, Shinichi and Kent Smetters, "Does Social Security Privatization Produce Efficiency Gains?," *Quarterly Journal of Economics*, 122:4, pp. 1677-1719 (November 2007).
* Peterman, William, "Reconciling Micro and Macro Estimates of the Frisch Labor Supply Elasticity," *Economic Inquiry*, 54:1, pp. 100-120 (January 2016).
* Weil, Philippe, "Overlapping Generations: The First Jubilee," *Journal of Economic Perspectives*, 22(4), 115-134 (Fall 2008).


## Disability services ##

If you need any special accommodations, please provide us with a copy of your Accommodation Determination Letter (provided to you by the Student Disability Services office) as soon as possible so that you may discuss with me how your accommodations may be implemented in this course.
