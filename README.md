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

The parameter <img src="https://latex.codecogs.com/svg.image?\alpha"/> controls the variance decay rate. Denoting<img src="https://latex.codecogs.com/svg.image?d\phi_i=\frac{1}{\alpha}\frac{d\phi_i}{dt}" style="vertical-align:middle"/>, then from time <img src="https://latex.codecogs.com/svg.image?t"/> to <img src="https://latex.codecogs.com/svg.image?t+\delta&space;t"/>, one has:

<img src="https://latex.codecogs.com/svg.image?\begin{align*}Var(t&plus;\delta&space;t)-Var(t)&=\frac{1}{N}\sum_i^N(\phi_i&plus;\alpha&space;d\phi_i\delta&space;t-\bar\phi)^2&space;-&space;\frac{1}{N}\sum_i^N(\phi_i-\bar\phi)^2\\&=\frac{1}{N}\sum_i^N\left[2\alpha&space;d\phi_i\delta&space;t(\phi_i-\bar\phi)&space;&plus;&space;\alpha^2&space;d\phi_i^2&space;\delta&space;t^2\right]\\&=2\alpha&space;\delta&space;t&space;\langle&space;\phi&space;\cdot&space;d\phi&space;\rangle&space;&plus;&space;\alpha^2\delta&space;t^2\langle&space;d\phi^2\rangle\end{align*}" title="\begin{align*}Var(t+\delta t)-Var(t)&=\frac{1}{N}\sum_i^N(\phi_i+\alpha d\phi_i\delta t-\bar\phi)^2 - \frac{1}{N}\sum_i^N(\phi_i-\bar\phi)^2\\&=\frac{1}{N}\sum_i^N\left[2\alpha d\phi_i\delta t(\phi_i-\bar\phi) + \alpha^2 d\phi_i^2 \delta t^2\right]\\&=2\alpha \delta t \langle \phi \cdot d\phi \rangle + \alpha^2\delta t^2\langle d\phi^2\rangle\end{align*}" />

And by variance decay rule of micro-mixing model:

<img src="https://latex.codecogs.com/svg.image?Var(t&plus;\delta&space;t)-Var(t)=\left[e^{-\Omega_\phi\delta&space;t}-1\right]Var(t)\approx-\Omega_\phi\delta&space;tVar(t)" title="Var(t+\delta t)-Var(t)=\left[e^{-\Omega_\phi\delta t}-1\right]Var(t)\approx-\Omega_\phi\delta tVar(t)" />

Thus, following the variance decay rule, we need to let <img src="https://latex.codecogs.com/svg.image?\alpha"/> satisfy the equation

<img src="https://latex.codecogs.com/svg.image?\begin{align*}\alpha^2\delta&space;t\langle&space;d\phi^2\rangle&plus;2\alpha\langle\phi\cdot&space;d\phi\rangle&plus;\Omega_\phi\langle\phi''^2\rangle=0\end{align*}" title="\begin{align*}\alpha^2\delta t\langle d\phi^2\rangle+2\alpha\langle\phi\cdot d\phi\rangle+\Omega_\phi\langle\phi''^2\rangle=0\end{align*}" />

In general, <img src="https://latex.codecogs.com/svg.image?\inline&space;A\equiv\delta&space;t\langle&space;d\phi^2\rangle>0" title="\inline A\equiv\delta t\langle d\phi^2\rangle>0" />,<img src="https://latex.codecogs.com/svg.image?\inline&space;B\equiv2\langle\phi\cdot&space;d\phi\rangle<0" title="\inline B\equiv2\langle\phi\cdot d\phi\rangle<0" />, <img src="https://latex.codecogs.com/svg.image?\inline&space;C\equiv\Omega_\phi&space;\langle\phi''^2\rangle>0" title="\inline C\equiv\Omega_\phi \langle\phi''^2\rangle>0" />. To get at least one real number solution, one need:

<img src="https://latex.codecogs.com/svg.image?B^2-4AC\ge0\Rightarrow\delta&space;t\le\frac{\langle\phi\cdot&space;d\phi\rangle}{\Omega_\phi\langle&space;d\phi^2\rangle\langle\phi''^2\rangle}" title="B^2-4AC\ge0\Rightarrow\delta t\le\frac{\langle\phi\cdot d\phi\rangle}{\Omega_\phi\langle d\phi^2\rangle\langle\phi''^2\rangle}" />

Once <img src="https://latex.codecogs.com/svg.image?\delta&space;t"/> is selected, the mixing ratio <img src="https://latex.codecogs.com/svg.image?\alpha"/> is hence to be:

<img src="https://latex.codecogs.com/svg.image?\alpha=\frac{-B&plus;\sqrt{B^2-4AC}}{2A}" title="\alpha=\frac{-B+\sqrt{B^2-4AC}}{2A}" />

For multiple scalars, single <img src="https://latex.codecogs.com/svg.image?\alpha"/> could not lead to target variance decay for all scalars. So a root finding technique is employed to estimate best <img src="https://latex.codecogs.com/svg.image?\alpha"/>. 
$$
\begin{align*}
\alpha_{new} = \alpha_{old} \left(1-e^{-\Omega_\phi\delta t}\right)/\overline{\left(1-\frac{Var(\phi+\alpha_{old} d\phi\delta t)}{Var(\phi)}\right)}
\end{align*}
$$
Typically, the root finding process converges in 2-3 loops.




#### 4. References

+ PaSR code: https://github.com/SLACKHA/pyJac/blob/master/pyjac/functional_tester/partially_stirred_reactor.py
+ EMST code: https://tcg.mae.cornell.edu/emst/
+ EMST paper: S. Subramaniam, S.B. Pope, A mixing model for turbulent reactive flows based on Euclidean minimum spanning trees, *Combust. Flame* 115 (1998) 487-514.
