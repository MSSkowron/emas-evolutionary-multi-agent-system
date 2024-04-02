# Computational intelligence

## EMAS (Evolutionary Multi-agent System)

    https://www.age.agh.edu.pl/agent-based-computing/emas-2/

## irace (Iterated Racing for Automatic Algorithm Configuration) - Lopez-Ibanez, Stutzle

## Plan:

1. **Zmiana operatorów**

    Wykorzystać istniejące implementacje operatorów (mutacja, krzyżowanie).
    Można skorzystać z gotowych rozwiązań z jmetal, takich jak sbx (https://github.com/jMetal/jMetalPy/tree/main/jmetal/operator).
    Warto również rozważyć bibliotekę DEAP (https://deap.readthedocs.io/en/master/).
    Operatory powinny być stosowane zgodnie z podejściem "pipe and filters".

2. ~~**Akcje**~~

    ~~W każdej iteracji jest wywoływane po kolei:~~

   ~~1. - Reprodukcja~~ 
   
    ~~Dla każdego osobnika losujemy z pozostałych osobników partnera do reprodukcji.
     Reprodukcja wymaga jakiejś minimalnej energii, która jest parametryzowalna (tak jak mieliśmy wcześniej).
     Reprodukcja zabiera rodzicom troche ich energii, która jest przekazywana dziecku.~~

   ~~2. - Walka~~
   
     ~~Dla każdego osobnika losujemy z pozostałych osobników przeciwnika do walki.
     W wyniku walki nastepuje przepływ energii jeden traci drugi zyskuje (energia w układzie musi być stała!)~~

   ~~3. - Sprzątanie zmarłych~~
    
3. ~~**Usunięcie migracji**~~

    ~~Na początkowym etapie ograniczyć się do pracy na pojedynczej "wyspie".~~

4. **Reprezentacja osobnika jako mającego dyskretną energię i genotyp będący wektorem**

    Kontynuować z wykorzystaniem wektorowej reprezentacji osobnika (np. Agent.x jako wektor współrzędnych x1, x2, ..., xn). Usunąć y, który mamy w klasie `Agent`. Ma być tylko wektor x i energia.

5. **Dyskretna i stała energia**

    Utrzymać dyskretną i stałą wartość energii przez całą symulację.
    W trakcie walki zachować pierwotną logikę przekazywania energii.

6. **Zgeneralizowanie dla różnych funkcji i wymiarów**

    Planować algorytm tak, aby działał dla różnych funkcji celu i różnych wymiarów problemu.

7. ~~**Wykres**~~

    ~~Na osi x umieścić kolejne iteracje.
    Na osi y przedstawiać najlepszego agenta (z najwyższą energią) w danej iteracji, czyli minimum, które zostało znalezione do tej pory.~~
