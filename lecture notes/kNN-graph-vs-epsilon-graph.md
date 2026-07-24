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

$$k \approx n \cdot f(x) \cdot V_d \cdot r_k(x)^d \quad\Longrightarrow\quad r_k(x) \approx \left(\frac{k}{n \cdot f(x) \cdot V_d}\right)^{1/d},$$

donde $V_d$ es el volumen de la bola unitaria. Es decir, $r_k(x)$ **se encoge donde los datos son densos y crece donde son dispersos**, compensando automáticamente las variaciones de densidad — exactamente el comportamiento observado con las dos lunas: un $\varepsilon$ fijo, o bien une las lunas por un punto de ruido, o bien fragmenta las zonas más dispersas, mientras que $r_k(x)$ se reescala localmente.

### Apéndice: derivación detallada

Sean $X_1,\dots,X_n \stackrel{\text{iid}}{\sim} f$ en $\mathbb{R}^d$, y fijemos un punto $x$.

**Paso 1 — La densidad como una tasa de probabilidad local.** Por definición de densidad, para una bola $B(x,r)$:

$$P\big(X_i \in B(x,r)\big) = \int_{B(x,r)} f(u)\,du.$$

Si $f$ es continua en $x$ y $r$ es pequeño, $f$ es aproximadamente constante sobre la bola, así que la integral se aproxima por el valor constante $f(x)$ multiplicado por el volumen de la bola:

$$P\big(X_i \in B(x,r)\big) \;\approx\; f(x)\cdot \operatorname{Vol}\big(B(x,r)\big) = f(x)\cdot V_d \cdot r^d,$$

donde $V_d = \pi^{d/2}/\Gamma(d/2+1)$ es el volumen de la bola unitaria en $\mathbb{R}^d$ (p. ej. $V_1=2$, $V_2=\pi$, $V_3=\tfrac{4}{3}\pi$). Esta es la aproximación de primer orden "densidad $\times$ volumen $\approx$ probabilidad", exacta en el límite $r\to 0$.

**Paso 2 — Contar puntos dentro de la bola.** Sea $N(r)$ el número de los $n$ puntos de la muestra que caen en $B(x,r)$. Como cada $X_i$ cae en la bola de forma independiente con probabilidad $p(r):=P(X_i\in B(x,r))$,

$$N(r) \sim \text{Binomial}\big(n,\ p(r)\big), \qquad \mathbb{E}[N(r)] = n\,p(r) \approx n\,f(x)\,V_d\,r^d.$$

**Paso 3 — Qué significa $r_k(x)$.** Por definición, $r_k(x)$ es exactamente el radio en el que la bola contiene por primera vez $k$ puntos de la muestra, es decir, el $k$-ésimo estadístico de orden de las distancias $\{d(x,X_i)\}$, caracterizado por

$$N\big(r_k(x)\big) = k.$$

Esta es una afirmación *exacta* sobre la muestra observada, todavía no una aproximación.

**Paso 4 — El paso heurístico: reemplazar el conteo por su esperanza.** La aproximación consiste en tratar el conteo observado $k$ en el radio $r_k(x)$ como si fuera el conteo *esperado* en ese mismo radio:

$$k = N\big(r_k(x)\big) \;\approx\; \mathbb{E}\big[N(r_k(x))\big] \;\approx\; n\,f(x)\,V_d\,r_k(x)^d.$$

**Por qué este reemplazo se justifica (argumento de concentración).** Localmente, cerca de $x$, un proceso Binomial($n$, $p$ pequeño) se aproxima bien por un **proceso de Poisson** de intensidad $\lambda(x)=n f(x)$ puntos por unidad de volumen (límite de Poisson del Binomial cuando $n\to\infty$, $p\to 0$, $np\to$ cte). Para un proceso de Poisson homogéneo de tasa $\lambda$, el volumen $\lambda\cdot V_d\cdot r_k(x)^d$ necesario para capturar $k$ puntos es la suma de $k$ "huecos" exponenciales i.i.d. de tasa unitaria, es decir, tiene distribución $\text{Gamma}(k,1)$, con

$$\mathbb{E}\big[\lambda V_d r_k(x)^d\big] = k, \qquad \operatorname{sd}\big[\lambda V_d r_k(x)^d\big] = \sqrt{k}.$$

Así, la fluctuación *relativa* de $\lambda V_d r_k(x)^d$ alrededor de $k$ es $O(1/\sqrt{k})$: por la ley de los grandes números se concentra fuertemente en torno a $k$ cuando $k\to\infty$. Esto es lo que justifica reemplazar la cantidad aleatoria $N(r_k(x))=k$ por su media $n f(x) V_d r_k(x)^d$: la aproximación es buena precisamente cuando $k$ es grande (muchos vecinos) y a la vez $k/n\to 0$ (de modo que $r_k(x)\to 0$ y también mejora la aproximación de densidad localmente constante del Paso 1).

**Paso 5 — Despejar $r_k(x)$.**

$$n\,f(x)\,V_d\,r_k(x)^d \approx k \quad\Longrightarrow\quad r_k(x)^d \approx \frac{k}{n\cdot f(x)\cdot V_d} \quad\Longrightarrow\quad r_k(x) \approx \left(\frac{k}{n\cdot f(x)\cdot V_d}\right)^{1/d}.$$

**Observación — esto es el estimador clásico de densidad por $k$-vecinos, invertido.** Despejando la misma ecuación para $f(x)$ en vez de $r_k(x)$ se obtiene

$$\hat f(x) = \frac{k}{n\cdot V_d\cdot r_k(x)^d},$$

que es el estimador de Loftsgaarden–Quesenberry (1965) por $k$-vecinos más cercanos: se usa el $r_k(x)$ *observado* para estimar la densidad desconocida. Esta nota recorre la misma relación al revés —suponiendo $f(x)$ fija/conocida, predice cómo escala $r_k(x)$— para mostrar cualitativamente que $r_k(x) \propto f(x)^{-1/d}$: el radio de $k$-vecinos se encoge en zonas densas y crece en zonas dispersas.

**Caso especial:** si $f$ es (localmente) constante, $r_k(x)\approx r$ para todo $x$, así que $\varepsilon_{\min}\approx\varepsilon_{\max}\approx r$ y $G_k(X)\approx G_r(X)$ — las dos construcciones coinciden exactamente cuando la densidad es uniforme. Este es también el puente conceptual hacia construcciones como las **filtraciones DTM** (distance-to-measure) en TDA, que reemplazan $d(x,y)$ por algo como $\max(d(x,y), r_k(x), r_k(y))$ precisamente para obtener una filtración tipo Rips adaptada a la densidad y robusta al ruido.
