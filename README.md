# User Modelling for Avoiding Overfitting in Interactive Knowledge Elicitation for Prediction

This repository contains the codes and data for user modeling to avoid user overfitting in interactive machine learning. Many algorithms and user interfaces often expose the user to the training data or its statistics which may lead to double use of data and overfitting, if the user reinforces noisy patterns in the data. We propose a user modelling methodology, by assuming simple rational behaviour, to correct this problem [1].

We apply our approach to infer user knowledge on feature relevance (probability of relevance of words in the Amazon reviews) in sparse linear regression. We use a probabilistic sparse linear regression model described in [Daee, P., Peltola, T., Soare, M. et al. Mach Learn (2017) 106: 1599. https://doi.org/10.1007/s10994-017-5651-7].

## Data and user study 

[Data-Exp1](Data-Exp1) contains the explanations of each experiment, data, and the user responses.

[main.m](main.m) runs the user study and generates the results. The user model is implemented in this script (in Method "User FB after correction").

[word_analysis.m](word_analysis.m) compares the user feedbacks in the two system.

[linreg_sns_ep.m](linreg_sns_ep.m) performs the posterior approximation (using Expectation Propagation).


## Citation

If you are using this source code in your research please consider citing us:

[1] Pedram Daee, Tomi Peltola, Aki Vehtari and Samuel Kaski. **User Modelling for Avoiding Overfitting in Interactive Knowledge Elicitation for Prediction**, In Proceedings of the the 23rd ACM International Conference on Intelligent User Interfaces (IUI 2018) (to appear). arXiv preprint arXiv:1710.04881 (2018). [[preprint](https://arxiv.org/abs/1710.04881)] [reviews].


## Team

[![Pedram Daee](https://sites.google.com/site/pedramdaee/_/rsrc/1428612543885/home/Pedram.jpg?height=100&width=76)](https://github.com/PedramDaee) | [![Tomi Peltola](http://research.cs.aalto.fi/pml/personnelpics/tomi.jpg?s=500)](https://github.com/to-mi) 
---|---
[Pedram Daee](https://sites.google.com/site/pedramdaee/home) | [Tomi Peltola](https://github.com/to-mi) 


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details