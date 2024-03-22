# Computational intelligence

## EMAS (Evolutionary Multi-agent System)

- https://www.age.agh.edu.pl/agent-based-computing/emas-2/

## irace (Iterated Racing for Automatic Algorithm Configuration) - Lopez-Ibanez, Stutzle


## Plan:
jmetal:
    sbx
    https://github.com/jMetal/jMetalPy/tree/main/jmetal/operator

może coś takiego:
https://deap.readthedocs.io/en/master/

1 wyspa tylko, bez migracji
mutacja i krzyżowanie tylko z operatorów, zaprogramować jako pipe and filters
poprawić mutacje jakąs bardziej zaawansowaną, ale jakąś gotową skądś (najelpiej wrócić do tego co było dzielenie wektorów w krzyżownaniu)

rozszerzyć rozwiązanie na wiele wymiarów:
    osobnik reprezentowany jako wektor
walka:
    poprzez losowanie uczestników
    przekazywanie energii
każda operator wymaga jakiejś energii, przekazywanie energii
dyskretna energia i stała w systemie
zgeneralizować dla różnych funkcji i wielu wymiarów

wykresy:
    na osi x iteracja
    na osi y najlepszy agent, z najwyższą energią

