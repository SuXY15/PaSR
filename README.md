## PaSR

Partially Stirred Reactor using Cantera for validating mixing models.

#### 1. Dependencies

```
cantera >= 2.5
mlpack
networkx
```



#### 2. Part of results

+ PaSR simulation results of H2/N2-Air mixture, with IEM, MC and EMST as mixing model.

​	<img src="figs/IEM_Z-T.png" style="width:30%;" /> <img src="figs/MC_Z-T.png" style="width:30%;" /> <img src="figs/EMST_Z-T.png" style="width:30%;" />

+ PaSR simulation results of H2/N2-Air mixture, with KerM as mixing model of different kernel sizes.

  <img src="figs/KerM10_Z-T.png" style="width:30%;" /> <img src="figs/KerM03_Z-T.png" style="width:30%;" /> <img src="figs/KerM01_Z-T.png" style="width:30%;" />



#### 3. Implementation details

