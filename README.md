# ECO
Ecological and evolutionary simulations

## Disclaimer
The project is still in development, use code at your own risk!

## History
In order to simulate disruptive selection, a multi-peaked environmental fitness landscape function was designed. Individuals have hermaphroditic sexual reproduction and chromosomes subject to crossover. A minimal mutation rate is also present. The chances of reproduction are determined by their individual relative fitness, given by the fitness landscape on the phenotypic value of the individuals.


We note that disruptive selection cannot occur by the action of a multi-peaked environmental fitness landscape alone in a panmictic population. More acting forces are needed to be present with some specific features.

1. Trait-dependent sexual preference, which for this case should make similar phenotypes mutually attractive. 
2. Trait-dependent intraspecific competition, which depends on the phenotype probability density function (PDF). E.g. resource extinction: fish size determines its prey, so if there are too many of similar sizes, they will have to strive harder to survive


![A](gallery/disrupt_fail.png "A")
![B](gallery/disrupt_succ.png "B")
