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

  <img src="figs/IEM_Z-T.png" style="width:32%;" /> <img src="figs/MC_Z-T.png" style="width:32%;" /> <img src="figs/EMST_Z-T.png" style="width:32%;" />

+ PaSR simulation results of H2/N2-Air mixture, with KerM as mixing model of different kernel sizes.

  <img src="figs/KerM10_Z-T.png" style="width:32%;" /> <img src="figs/KerM03_Z-T.png" style="width:32%;" /> <img src="figs/KerM01_Z-T.png" style="width:32%;" />



#### 3. Implementation details

##### 3.1 EMST variance decay

The mixing happens on the edges of EMST tree, with:

<img src="https://latex.codecogs.com/svg.image?\frac{d\phi_i}{dt}=-\alpha\frac{1}{w_i}\sum_v^{N_T-1}B_v\{(\phi_i-\phi_{n_v})\delta_{im_v}+(\phi_i-\phi_{m_v})\delta_{in_v}\}"/>

The parameter $\alpha$ controls the variance decay rate. Denoting <img src="https://latex.codecogs.com/svg.image?d\phi_i=\frac{1}{\alpha}\frac{d\phi_i}{dt}" />, then from time <img src="https://latex.codecogs.com/svg.image?t"/> to <img src="https://latex.codecogs.com/svg.image?t+\delta t"/>, one has:

<img src="https://latex.codecogs.com/svg.image?\begin{align*}Var(t&plus;\delta&space;t)-Var(t)&=\frac{1}{N}\sum_i^N(\phi_i&plus;\alpha&space;d\phi_i\delta&space;t-\bar\phi)^2&space;-&space;\frac{1}{N}\sum_i^N(\phi_i-\bar\phi)^2\\&=\frac{1}{N}\sum_i^N\left[2\alpha&space;d\phi_i\delta&space;t(\phi_i-\bar\phi)&space;&plus;&space;\alpha^2&space;d\phi_i^2&space;\delta&space;t^2\right]\\&=2\alpha&space;\delta&space;t&space;\langle&space;\phi&space;\cdot&space;d\phi&space;\rangle&space;&plus;&space;\alpha^2\delta&space;t^2\langle&space;d\phi^2\rangle\end{align*}" title="\begin{align*}Var(t+\delta t)-Var(t)&=\frac{1}{N}\sum_i^N(\phi_i+\alpha d\phi_i\delta t-\bar\phi)^2 - \frac{1}{N}\sum_i^N(\phi_i-\bar\phi)^2\\&=\frac{1}{N}\sum_i^N\left[2\alpha d\phi_i\delta t(\phi_i-\bar\phi) + \alpha^2 d\phi_i^2 \delta t^2\right]\\&=2\alpha \delta t \langle \phi \cdot d\phi \rangle + \alpha^2\delta t^2\langle d\phi^2\rangle\end{align*}" />

And by variance decay rule of micro-mixing model:
$$
Var(t+\delta t)-Var(t) = \left[e^{-\Omega_\phi\delta t}-1\right]Var(t)  \approx -\Omega_\phi \delta t Var(t)
$$
Thus, following the variance decay rule, we need to let <img src="https://latex.codecogs.com/svg.image?\alpha"/> satisfy the equation
$$
\begin{align*}
\alpha^2\delta t\langle d\phi^2\rangle + 2\alpha \langle \phi \cdot d\phi \rangle + \Omega_\phi \langle\phi''^2\rangle = 0
\end{align*}
$$
In general, $A\equiv\delta t\langle d\phi^2\rangle>0$, $B\equiv 2 \langle \phi \cdot d\phi \rangle <0$, $C\equiv\Omega_\phi \langle\phi''^2\rangle>0$. To get at least one real number solution, one need:
$$
B^2-4AC\ge0 \Rightarrow \delta t\le \frac{\langle\phi\cdot d\phi\rangle}{\Omega_\phi\langle d\phi^2\rangle\langle\phi''^2\rangle}
$$
Once $\delta t$ is selected, the mixing ratio $\alpha$ is hence to be:
$$
\alpha=\frac{-B+\sqrt{B^2-4AC}}{2A}
$$






#### 4. References

+ PaSR: https://github.com/SLACKHA/pyJac/blob/master/pyjac/functional_tester/partially_stirred_reactor.py
+ EMST: https://tcg.mae.cornell.edu/emst/
