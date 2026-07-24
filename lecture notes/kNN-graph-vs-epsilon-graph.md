# Relación matemática entre el grafo de $k$-vecinos y el grafo de $\varepsilon$-vecindades

Nota complementaria a `UNAL_Bog_TDA_Clase_2_Agrupamientos_Clustering.ipynb`.

## Definiciones

Sea $(X,d)$ un espacio métrico finito (la nube de puntos), $|X| = n$.

**Grafo de $\varepsilon$-vecindades.** Se fija una única escala global $\varepsilon > 0$:

$$G_\varepsilon(X) = (X, E_\varepsilon), \qquad E_\varepsilon = \{\{x,y\} : d(x,y)\le \varepsilon\}.$$

Este grafo es exactamente el 1-esqueleto del complejo de Vietoris–Rips $VR_\varepsilon(X)$.

**Grafo de $k$-vecinos más cercanos.** Para cada $x\in X$, sea $N_k(x)$ el conjunto de sus $k$ puntos más cercanos y

$$r_k(x) := \max_{y\in N_k(x)} d(x,y)$$

la distancia al $k$-ésimo vecino más cercano de $x$. El grafo **dirigido** de $k$-vecinos tiene una arista $x\to y$ si y solo si $y\in N_k(x)$, es decir:

$$x\to y \iff d(x,y)\le r_k(x).$$

## La identidad clave

$N_k(x) = B(x,r_k(x))\setminus\{x\}$: una bola de radio $\varepsilon$ centrada en $x$, pero con **$\varepsilon$ reemplazado por el radio $r_k(x)$, que depende del punto**. Es decir:

> **El grafo de $k$-vecinos es un grafo de $\varepsilon$-vecindades cuyo radio es una función $r_k:X\to\mathbb{R}_{>0}$ en lugar de una constante.**

La función `connected_components(..., directed=False)` de `scipy` simetriza tomando la **unión** ("OR") de las dos direcciones:

$$\{x,y\}\in E_k \iff d(x,y)\le r_k(x) \ \text{ o }\ d(x,y)\le r_k(y) \iff d(x,y)\le \max(r_k(x),r_k(y)).$$

(La variante "mutua" del grafo de $k$-vecinos exige en cambio $d(x,y)\le \min(r_k(x),r_k(y))$.)

## Cota tipo "sándwich"

Sean $\varepsilon_{\min}=\min_x r_k(x)$ y $\varepsilon_{\max}=\max_x r_k(x)$. Entonces, como conjuntos de aristas sobre $X$:

$$G_{\varepsilon_{\min}}(X) \ \subseteq\ G_k^{\text{mutuo}}(X)\ \subseteq\ G_k^{\text{unión}}(X)\ \subseteq\ G_{\varepsilon_{\max}}(X).$$

**Demostración.** Si $d(x,y)\le\varepsilon_{\min}$, entonces $d(x,y)\le \min(r_k(x),r_k(y))$, así que la arista está en el grafo mutuo, y por lo tanto en el grafo unión. Si una arista está en el grafo unión, $d(x,y)\le\max(r_k(x),r_k(y))\le\varepsilon_{\max}$, así que también está en $G_{\varepsilon_{\max}}$. $\blacksquare$

Así, el grafo de $k$-vecinos siempre queda "atrapado" entre dos grafos de $\varepsilon$-vecindades ordinarios — pero el valor de $\varepsilon$ que se usa **en cada punto** lo elige el propio conjunto de datos.

## Ambos son filtraciones, pero en parámetros distintos

- $\varepsilon\le\varepsilon' \implies G_\varepsilon(X)\subseteq G_{\varepsilon'}(X)$ — familia anidada en $\varepsilon$ (es literalmente la filtración de Rips usada en homología persistente).
- $k\le k' \implies N_k(x)\subseteq N_{k'}(x) \implies G_k(X)\subseteq G_{k'}(X)$ — familia anidada en $k$.

Ambas familias degeneran en los mismos extremos: $k=1$ (o $\varepsilon\to 0^+$) da el grafo más disperso, y $k=n-1$ (o $\varepsilon\to\infty$) da el grafo completo.

## Por qué $r_k(x)$ se adapta a la densidad

Si $X\sim f$ i.i.d. en $\mathbb{R}^d$, el estimador de densidad por $k$-vecinos da

$$k \approx n\, f(x)\, V_d\, r_k(x)^d \quad\Longrightarrow\quad r_k(x) \approx \left(\frac{k}{n\,f(x)\,V_d}\right)^{1/d},$$

donde $V_d$ es el volumen de la bola unitaria. Es decir, $r_k(x)$ **se encoge donde los datos son densos y crece donde son dispersos**, compensando automáticamente las variaciones de densidad — exactamente el comportamiento observado con las dos lunas: un $\varepsilon$ fijo, o bien une las lunas por un punto de ruido, o bien fragmenta las zonas más dispersas, mientras que $r_k(x)$ se reescala localmente.

**Caso especial:** si $f$ es (localmente) constante, $r_k(x)\approx r$ para todo $x$, así que $\varepsilon_{\min}\approx\varepsilon_{\max}\approx r$ y $G_k(X)\approx G_r(X)$ — las dos construcciones coinciden exactamente cuando la densidad es uniforme. Este es también el puente conceptual hacia construcciones como las **filtraciones DTM** (distance-to-measure) en TDA, que reemplazan $d(x,y)$ por algo como $\max(d(x,y), r_k(x), r_k(y))$ precisamente para obtener una filtración tipo Rips adaptada a la densidad y robusta al ruido.
