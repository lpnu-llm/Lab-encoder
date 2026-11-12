# %% [markdown]
# # Transformer encoder
#
# ## Skip connection: як навчати глибоку мережу
#
# У попередніх роботах ми реалізували attention та позиційне кодування.
# Перед складанням encoder залишилась ще одна деталь: **залишковий зв'язок**
# (residual / skip connection). Він переносить вхід повз підмережу й додає
# його до її результату. Спочатку працюємо зі списками та числами Python,
# потім перевіряємо backprop вручну й досліджуємо границю великої глибини.
#
# Заповнюйте `...` і запускайте перевірки після кожної вправи.
# Основна частина використовує лише стандартну бібліотеку Python.

# %% [markdown]
# ## 1. Вчимо поправку до входу
#
# Звичайний блок обчислює $y=F(x)$. Залишковий блок обчислює
# $$y=x+\alpha F(x).$$
# Тут $x$ — shortcut (обхідний шлях), $F$ — навчувана гілка, а $\alpha$ —
# масштаб поправки (спочатку 1). Якщо бажане перетворення $H(x)$ близьке
# до входу, гілка має вивчити лише $F(x)=H(x)-x$ при $\alpha=1$.
# Для тотожного перетворення достатньо $F(x)=0$. Додавання блоків із
# нульовою гілкою тому не змінює функцію мережі. Це зручна параметризація
# для оптимізації, а не гарантія кращого результату на будь-якій задачі.
#
# ![Прямий та зворотний шляхи залишкового блока](figures/residual.svg)
#
# Реалізуйте `residual_add`: поелементне додавання двох векторів однакової
# довжини. Не змінюйте входи; при різній довжині піднімайте `ValueError`.
# Це додавання, а не конкатенація: кількість ознак залишається сталою.

# %%
from math import exp, isclose


def residual_add(x, branch, alpha=1.0):
    """Повертає x + alpha * branch для двох списків однакової довжини."""
    ...


def test_residual_add():
    x, branch = [1.0, -2.0], [0.5, 3.0]
    assert residual_add(x, branch) == [1.5, 1.0]
    assert residual_add(x, branch, 0.1) == [1.05, -1.7]
    assert residual_add(x, [0.0, 0.0]) == x
    assert residual_add(x, branch, 0.0) == x
    assert residual_add([], []) == []
    assert x == [1.0, -2.0] and branch == [0.5, 3.0]
    try:
        residual_add([1.0], [1.0, 2.0])
    except ValueError:
        pass
    else:
        raise AssertionError("Перевірте форму обох гілок")


if __name__ == "__main__":
    test_residual_add()
    print("✓ residual_add")

# %% [markdown]
# **Приклад.** Якщо $x=[1,-2]$, а гілка повертає $[0.5,3]$, то вихід —
# $[1.5,1]$. Вхід не загубився: наступний блок отримує його разом із поправкою.
# Для embedding-ів форми `(batch, tokens, d_model)` обидві гілки повинні
# мати цю саму форму. Якщо розмірність змінюється, потрібна проєкція
# $P(x)+F(x)$; похідна shortcut тоді $J_P$, а не одинична матриця.
#
# Питання: чому нульова звичайна гілка знищує вхід, а нульова залишкова — ні?

# %% [markdown]
# ## 2. Чому глибина впливає на backpropagation
#
# Backpropagation застосовує правило ланцюга. Позначимо
# $g_l=\partial\mathcal L/\partial x_l$. Для скалярного звичайного блока
# $x_{l+1}=F_l(x_l)$ отримуємо $g_l=g_{l+1}F'_l(x_l)$.
# Для залишкового блока:
# $$g_l=g_{l+1}\left(1+\alpha F'_l(x_l)\right).$$
# Градієнт від додавання надходить до **обох** гілок; у вході внески
# додаються. Векторна версія для градієнтів-стовпців:
# $$g_l=(I+\alpha J_{F_l})^Tg_{l+1}.$$
# Член $I$ дає прямий шлях сигналу похибки. Проте $J_F=-I$ при
# $\alpha=1$ обнуляє весь вираз, а великі додатні похідні його збільшують.
# Отже shortcut не гарантує ненульових чи обмежених градієнтів.
#
# Візьмемо $F_l(x)=w_lx$. Реалізуйте один блок та його backward.
# `upstream` — $\partial\mathcal L/\partial y$, результат — пара
# $(\partial\mathcal L/\partial x,\partial\mathcal L/\partial w)$.
# У залишковому випадку $\partial y/\partial w=\alpha x$;
# у звичайному — $x$. Масштаб `alpha` використовуємо лише для residual.

# %%
def scalar_block(x, w, residual=True, alpha=1.0):
    ...


def block_backward(x, w, upstream, residual=True, alpha=1.0):
    ...


def test_block_backward():
    assert scalar_block(2.0, 0.5) == 3.0
    assert scalar_block(2.0, 0.5, False) == 1.0
    assert block_backward(2.0, 0.5, 3.0) == (4.5, 6.0)
    assert block_backward(2.0, 0.5, 3.0, False) == (1.5, 6.0)
    dx, dw = block_backward(2.0, 0.5, 3.0, True, 0.1)
    assert isclose(dx, 3.15) and isclose(dw, 0.6)
    assert block_backward(2.0, -1.0, 1.0)[0] == 0.0
    # Незалежна чисельна перевірка обох похідних.
    eps = 1e-6
    for residual in (False, True):
        x, w, upstream, alpha = 0.7, -0.2, 1.3, 0.4
        dx, dw = block_backward(x, w, upstream, residual, alpha)
        def loss(a, b):
            return upstream * scalar_block(a, b, residual, alpha)
        numeric_x = (loss(x + eps, w) - loss(x - eps, w)) / (2 * eps)
        numeric_w = (loss(x, w + eps) - loss(x, w - eps)) / (2 * eps)
        assert isclose(dx, numeric_x, abs_tol=1e-8)
        assert isclose(dw, numeric_w, abs_tol=1e-8)


if __name__ == "__main__":
    test_block_backward()
    print("✓ backward одного блока")

# %% [markdown]
# ## 3. Мережа довільної скінченної глибини
#
# Збережемо всі $x_l$ під час forward. Для
# $\mathcal L=\tfrac12(x_N-t)^2$ початковий backward-сигнал — $x_N-t$.
# Пройдіть блоки у зворотному порядку. Кожна вага незалежна, навіть якщо
# початкові значення однакові. Поверніть loss, градієнт входу та список
# градієнтів ваг **у початковому порядку**. Поки що оптимізатор їх не оновлює.

# %%
def network_loss_grad(x, weights, target, residual=True, alpha=1.0):
    """Повертає (loss, dloss_dx, [dloss_dw0, ...])."""
    ...


def test_network_loss_grad():
    for residual in (False, True):
        x, weights, target, alpha = 0.7, [0.2, -0.3, 0.4], -0.1, 0.5
        loss, dx, dw = network_loss_grad(x, weights, target, residual, alpha)
        assert loss >= 0 and len(dw) == len(weights)
        eps = 1e-6
        def objective(a, ws):
            return network_loss_grad(a, ws, target, residual, alpha)[0]
        assert isclose(dx, (objective(x + eps, weights) - objective(x - eps, weights)) / (2 * eps), abs_tol=1e-8)
        for i in range(len(weights)):
            plus, minus = weights[:], weights[:]
            plus[i] += eps
            minus[i] -= eps
            assert isclose(dw[i], (objective(x, plus) - objective(x, minus)) / (2 * eps), abs_tol=1e-8)
        assert weights == [0.2, -0.3, 0.4]
    assert network_loss_grad(2.0, [], 1.0) == (0.5, 1.0, [])
    assert network_loss_grad(2.0, [0.0] * 100, 1.0)[:2] == (0.5, 1.0)


if __name__ == "__main__":
    test_network_loss_grad()
    print("✓ backward мережі")

# %% [markdown]
# ## 4. Експеримент: 1, 10, 100, 1000 шарів
#
# Щоб відокремити вплив мережі від впливу loss, виміряємо
# $\partial x_N/\partial x_0$ із кінцевим градієнтом **1**.
# Для однакових $w$ це $w^N$ без shortcut та $(1+\alpha w)^N$ із ним.
# Реалізуйте вимірювання циклом через `block_backward`, без степеня.
# Воно не потребує зберігати активації: у лінійному прикладі похідна за
# входом не залежить від активації. Загальна нелінійна мережа їх потребує.

# %%
def depth_gradient(depth, w, residual, alpha=1.0):
    """Повертає dx_N/dx_0; depth — невід'ємне ціле число."""
    ...


def test_depth_gradient():
    assert depth_gradient(0, 0.5, False) == 1.0
    assert isclose(depth_gradient(10, 0.5, False), 0.5 ** 10)
    assert depth_gradient(1000, 0.0, True) == 1.0
    assert depth_gradient(10, -1.0, True) == 0.0
    assert isclose(depth_gradient(10, 0.5, True, 0.1), 1.05 ** 10)


if __name__ == "__main__":
    test_depth_gradient()
    print(" N     plain w=.5    residual w=.5    residual w=0    residual w=-.5")
    for n in [1, 10, 100, 1000]:
        values = [depth_gradient(n, .5, False), depth_gradient(n, .5, True),
                  depth_gradient(n, 0., True), depth_gradient(n, -.5, True)]
        print(f"{n:4d} " + " ".join(f"{v:17.3e}" for v in values))

# %% [markdown]
# ![Залежність модуля градієнта від глибини](figures/depth.svg)
#
# У таблиці звичайна мережа з $w=0.5$ втрачає градієнт, а residual із тією
# самою вагою підсилює його як $1.5^N$. Нульова residual-гілка дає 1,
# а $w=-0.5$ знову дає $0.5^N$. Числа тут аналітичні, до навчання.
# Мала величина може округлитися до нуля (underflow), велика — до `inf`.
# Це обмеження чисел із рухомою комою, а не доказ математичного нуля.
#
# **Дослідіть:** змініть вагу на 0.01 і -0.01. Чи залишиться градієнт
# близьким до 1 при мільйоні шарів? Чому «близький до identity» для одного
# блока ще не означає стабільний добуток для довільної глибини?

# %% [markdown]
# ## 5. Що означає «нескінченна глибина»
#
# Нескінченний Python-цикл не завершиться. Натомість розглянемо сімейство
# мереж зі зростанням $N$, але **фіксованим загальним часом** $T$:
# $$x_{k+1}=x_k+\frac{T}{N}a x_k.$$
# Це метод Ейлера для $dx/dt=ax$ на $[0,T]$. У всіх блоків тут спільний
# коефіцієнт $a$: ми уточнюємо дискретизацію однієї динаміки, а не доводимо
# збіжність довільної мережі з незалежними параметрами.
#
# $$x_N=x_0(1+aT/N)^N\longrightarrow x_0e^{aT},\qquad
# \frac{\partial x_N}{\partial x_0}\longrightarrow e^{aT}.$$
# При $a=1,T=1$ масштаб $1/N$ дає границю $e$, тоді як без нього
# множник був би $2^N$. Для загальної гладкої гілки $F(x,t)$ аналогічний
# перехід веде до ODE; збіжність вимагає умов регулярності та стійкого кроку.
# Навіть $e^{aT}$ може бути дуже великим чи малим: scaling не усуває всі
# проблеми backpropagation. Для $a<0$ грубий крок Ейлера теж може бути нестійким.
#
# Реалізуйте `euler_limit` циклом; одночасно накопичуйте похідну за входом.
# Не використовуйте готову формулу чи `exp` у реалізації.

# %%
def euler_limit(x0, a, total_time, depth):
    """(Кінцевий стан, похідна за x0); depth — додатне ціле число."""
    ...


def test_euler_limit():
    assert euler_limit(2.0, 0.0, 1.0, 10) == (2.0, 1.0)
    assert euler_limit(2.0, 1.0, 0.0, 10) == (2.0, 1.0)
    errors = []
    for n in [10, 100, 1000, 10000]:
        state, grad = euler_limit(2.0, 1.0, 1.0, n)
        assert isclose(state, 2.0 * grad, rel_tol=1e-12)
        assert isclose(grad, (1.0 + 1.0 / n) ** n, rel_tol=1e-11)
        errors.append(abs(grad - exp(1)))
    assert all(new < old for old, new in zip(errors, errors[1:]))
    assert errors[-1] < 0.0002
    _, grad = euler_limit(1.0, -1.0, 1.0, 10000)
    assert isclose(grad, exp(-1), abs_tol=0.00002)


if __name__ == "__main__":
    test_euler_limit()
    print(" N       x_N          dx_N/dx_0     похибка градієнта")
    for n in [1, 10, 100, 1000, 10000]:
        state, grad = euler_limit(2.0, 1.0, 1.0, n)
        print(f"{n:5d}   {state:.8f}   {grad:.8f}   {abs(grad-exp(1)):.3e}")

# %% [markdown]
# **Як це змінює backprop?** У дискретній мережі множимо $N$ факторів
# $1+aT/N$. У неперервній границі чутливість $s(t)=\partial x(t)/\partial x_0$
# задовольняє $ds/dt=a s$, $s(0)=1$. Градієнт loss, який поширюється від
# кінця до початку, задовольняє $dg/dt=-a g$ з кінцевою умовою $g(T)$;
# звідси $g(0)=e^{aT}g(T)$.
#
# Пам'ять і час не стають безкоштовними: звичайний backprop зі збереженням
# активацій має витрати, що зростають із глибиною. Checkpointing обмінює
# пам'ять на повторні обчислення; ODE-solvers використовують скінченне число
# кроків із чисельною похибкою. Тут ми досліджували точну просту границю,
# а не реалізовували solver чи нескінченно глибокий Transformer.

# %% [markdown]
# ## 6. Місток до Transformer encoder
#
# Encoder має дві залишкові гілки: self-attention змішує інформацію токенів,
# а feed-forward (FFN) перетворює ознаки кожного токена окремо.
# У варіанті **pre-norm**:
# $$h=x+\operatorname{Attention}(\operatorname{LN}_1(x)),$$
# $$y=h+\operatorname{FFN}(\operatorname{LN}_2(h)).$$
# Shortcut обходить і підмережу, і LayerNorm. У початковому Transformer
# використовували **post-norm**: $h=\operatorname{LN}(x+F(x))$.
# Там похідна всього блока містить також Jacobian LayerNorm: чистий
# identity-шлях через весь блок уже не зберігається. Активація після
# додавання теж може перекрити градієнт, наприклад ReLU на від'ємному виході.
# Масштаб $1/N$ із попереднього розділу не є правилом для кожного encoder.
#
# Реалізуйте каркас pre-norm для **одного вектора** через передані функції.
# На наступній лабораторній замінимо їх справжніми підмережами для
# послідовності. У цій вправі тестові функції лише перевіряють порядок
# викликів; вони не реалізують attention чи LayerNorm.

# %%
def encoder_residuals(x, attention, ffn, norm1, norm2):
    ...


def test_encoder_residuals():
    calls = []
    def norm1(x):
        calls.append(("norm1", x[:]))
        return [v * 2 for v in x]
    def attention(x):
        calls.append(("attention", x[:]))
        return [1.0, -1.0]
    def norm2(x):
        calls.append(("norm2", x[:]))
        return [v * 3 for v in x]
    def ffn(x):
        calls.append(("ffn", x[:]))
        return [v * 0.5 for v in x]
    x = [1.0, 2.0]
    assert encoder_residuals(x, attention, ffn, norm1, norm2) == [5.0, 2.5]
    assert calls == [("norm1", [1.0, 2.0]), ("attention", [2.0, 4.0]),
                     ("norm2", [2.0, 1.0]), ("ffn", [6.0, 3.0])]
    assert x == [1.0, 2.0]


if __name__ == "__main__":
    test_encoder_residuals()
    print("✓ дві residual-гілки encoder")

# %% [markdown]
# ## 7. Додатково: той самий backward у PyTorch
#
# Після ручного виведення перевіримо, що autograd додає внески обох шляхів.
# Встановіть `torch` у середовище ноутбука (`%pip install torch` в окремій
# комірці). Цей розділ необов'язковий; без PyTorch він пропускається.
# `detach()` у shortcut обірвав би його градієнт — не використовуйте його.

# %%
if __name__ == "__main__":
    try:
        import torch
    except ImportError:
        print("Додатковий розділ пропущено: встановіть torch для autograd.")
    else:
        x = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
        weights = torch.tensor([0.2, -0.3, 0.4], dtype=torch.float64, requires_grad=True)
        h = x
        for w in weights:
            h = h + 0.5 * (w * h)
        loss = 0.5 * (h - (-0.1)) ** 2
        loss.backward()
        expected_loss, dx, dw = network_loss_grad(0.7, [0.2, -0.3, 0.4], -0.1, True, 0.5)
        assert isclose(loss.item(), expected_loss, abs_tol=1e-12)
        assert isclose(x.grad.item(), dx, abs_tol=1e-12)
        assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(weights.grad.tolist(), dw))
        print("✓ autograd збігається з ручним backward")

# %% [markdown]
# ## 8. Навіщо нормалізувати активації
#
# Після лінійних шарів і residual-додавань масштаб активацій може змінюватися.
# Великий масштаб, наприклад, може зробити attention-softmax дуже різким.
# Нормалізація керує масштабом входу наступної підмережі й часто полегшує
# оптимізацію. Вона не гарантує стабільності будь-якої мережі чи градієнтів.
# Це обчислення **всередині моделі**, а не лише підготовка датасету.
#
# Для групи чисел $z_1,\ldots,z_m$ визначимо:
# $$\mu=\frac1m\sum_i z_i,\quad v=\frac1m\sum_i(z_i-\mu)^2,\quad
# \hat z_i=\frac{z_i-\mu}{\sqrt{v+\epsilon}}.$$
# Ділимо дисперсію на $m$, а не $m-1$: тут нормалізуємо поточні активації.
# $\epsilon>0$ захищає від ділення на нуль і стоїть **під коренем**.
# До навчуваного перетворення середнє дорівнює нулю, а дисперсія —
# $v/(v+\epsilon)$, тобто лише приблизно 1 при $v\gg\epsilon$.
# Для сталої групи всі нормалізовані значення нульові.
#
# Наприклад, для `[1, 3]`: середнє 2, дисперсія 1, вихід приблизно `[-1, 1]`.
# Головне питання: **які числа об'єднуємо в одну групу?**

# %% [markdown]
# ### 8.1. BatchNorm: одна ознака в різних прикладах
#
# Для матриці `(batch, features)` BatchNorm обчислює статистики кожного
# **стовпця**: порівнює ту саму ознаку між прикладами батчу.
# Нехай батч має рядки `[1, 10]` і `[3, 14]`. Середні стовпців — `[2, 12]`,
# дисперсії — `[1, 4]`. Нормалізовані рядки приблизно `[-1, -1]`, `[1, 1]`.
#
# Після нормалізації застосовуємо $y_j=\gamma_j\hat x_j+\beta_j$.
# Вектори $\gamma$ та $\beta$ — навчувані параметри, зазвичай спочатку 1 і 0.
# Вони дозволяють оптимізатору обирати масштаб і зсув кожної ознаки.
# Статистики обчислюються з активацій; це не параметри, що навчаються SGD.
#
# **Training:** використовуємо поточний батч і накопичуємо running-оцінки.
# **Evaluation:** за звичайних налаштувань використовуємо збережені оцінки,
# тому результат не залежить від інших прикладів поточного inference-батчу.
# Наприклад, середнє оновлюється як
# `running_mean = (1-momentum)*running_mean + momentum*batch_mean`.
# У PyTorch running variance оновлюється незміщеною оцінкою (`m-1`), хоча
# training-forward нормалізує з дисперсією, поділеною на `m`.
#
# Готова демонстрація нижче приймає статистики явно; це **не повна реалізація
# BatchNorm** із накопиченням running-стану. Змінимо сусіда першого прикладу,
# а потім порівняємо training та evaluation з фіксованими статистиками.

# %%
def batch_norm_demo(rows, gamma, beta, mean=None, variance=None, eps=1e-5):
    """Непорожня прямокутна матриця; mean/variance задаються лише разом."""
    if (mean is None) != (variance is None):
        raise ValueError("Задайте обидві статистики")
    if mean is None:
        mean = [sum(column) / len(rows) for column in zip(*rows)]
        variance = [sum((row[j] - mean[j]) ** 2 for row in rows) / len(rows)
                    for j in range(len(mean))]
    return [[gamma[j] * (value - mean[j]) / (variance[j] + eps) ** 0.5 + beta[j]
             for j, value in enumerate(row)] for row in rows]


if __name__ == "__main__":
    batch_a = [[1.0, 10.0], [3.0, 14.0]]
    batch_b = [[1.0, 10.0], [0.0, 8.0]]
    for name, batch in [("A", batch_a), ("B", batch_b)]:
        train = batch_norm_demo(batch, [1, 1], [0, 0])
        evaluation = batch_norm_demo(batch, [1, 1], [0, 0], [2, 12], [1, 4])
        print(name, "перший рядок: train", train[0], "eval", evaluation[0])
    assert batch_norm_demo(batch_a, [1, 1], [0, 0])[0] != batch_norm_demo(batch_b, [1, 1], [0, 0])[0]
    assert batch_norm_demo(batch_a, [1, 1], [0, 0], [2, 12], [1, 4])[0] == batch_norm_demo(batch_b, [1, 1], [0, 0], [2, 12], [1, 4])[0]

# %% [markdown]
# BatchNorm широко застосовують у згорткових мережах; там статистики каналу
# агрегують також просторові позиції. Для послідовностей `BatchNorm1d` у
# PyTorch приймає `(batch, features, tokens)` і агрегує batch та tokens.
# Це інші осі, ніж наш простий двовимірний приклад.
# Малі батчі дають шумні оцінки; padding потребує уваги, бо може спотворювати
# статистики. Для Transformer зручно нормалізувати кожен токен окремо.

# %% [markdown]
# ## 9. LayerNorm: ознаки одного токена
#
# Для `(batch, tokens, d_model)` обчислюємо середнє та дисперсію вздовж
# **останньої осі**. Кожен токен отримує власні статистики:
# $$y_{b,t,j}=\gamma_j\frac{x_{b,t,j}-\mu_{b,t}}
# {\sqrt{v_{b,t}+\epsilon}}+\beta_j.$$
# Вектори $\gamma,\beta$ довжини `d_model` спільні для всіх токенів і
# прикладів у цьому шарі: всього `2*d_model` навчуваних параметрів.
# Два LayerNorm усередині блока мають окремі параметри.
#
# На відміну від BatchNorm, LayerNorm використовує поточні статистики
# токена і під час training, і під час evaluation; running-оцінок немає.
# Він працює і для batch size 1. Це не нормалізація довжини вектора до 1.
# LayerNorm не змішує інформацію токенів — цю роль виконує attention.
# Після довільних $\gamma,\beta$ вихід уже не зобов'язаний мати середнє 0.
#
# Реалізуйте `layer_norm` для одного непорожнього вектора. Вимагайте
# однакової довжини `x`, `gamma`, `beta` та `eps > 0`, інакше `ValueError`.
# Потім застосуйте його до списку токенів. Входи не змінюйте.

# %%
def layer_norm(x, gamma, beta, eps=1e-5):
    ...


def layer_norm_tokens(tokens, gamma, beta, eps=1e-5):
    """Матриця (tokens, d_model); порожня послідовність повертає []."""
    ...


def test_layer_norm():
    result = layer_norm([1.0, 3.0], [1.0, 1.0], [0.0, 0.0])
    expected = 1 / (1 + 1e-5) ** 0.5
    assert isclose(result[0], -expected) and isclose(result[1], expected)
    assert layer_norm([7.0, 7.0], [2, 3], [4, 5]) == [4.0, 5.0]
    assert layer_norm([9.0], [2], [3]) == [3.0]
    affine = layer_norm([1.0, 3.0], [2, 3], [4, 5])
    assert isclose(affine[0], 4 - 2 * expected)
    assert isclose(affine[1], 5 + 3 * expected)
    # Зсув усіх ознак токена зникає при центруванні.
    shifted = layer_norm([101.0, 103.0], [1, 1], [0, 0])
    assert all(isclose(a, b) for a, b in zip(result, shifted))
    for x, gamma, beta, eps in [([], [], [], 1e-5), ([1], [1, 2], [0], 1e-5),
                                ([1], [1], [], 1e-5), ([1], [1], [0], 0)]:
        try:
            layer_norm(x, gamma, beta, eps)
        except ValueError:
            pass
        else:
            raise AssertionError("Перевірте форми та epsilon")
    tokens = [[1.0, 10.0], [3.0, 14.0]]
    before = [row[:] for row in tokens]
    normalized = layer_norm_tokens(tokens, [1, 1], [0, 0])
    assert tokens == before
    assert layer_norm_tokens([], [1, 1], [0, 0]) == []
    changed = layer_norm_tokens([tokens[0], [1000.0, -20.0]], [1, 1], [0, 0])
    assert normalized[0] == changed[0]  # сусідній токен не впливає
    assert all(isclose(sum(row), 0.0, abs_tol=1e-12) for row in normalized)
    assert normalized[0][1] > 0  # BN першого прикладу мав би обидва значення < 0


if __name__ == "__main__":
    test_layer_norm()
    print("✓ LayerNorm: форми, статистики та незалежність токенів")
    print("BatchNorm:", batch_norm_demo([[1, 10], [3, 14]], [1, 1], [0, 0]))
    print("LayerNorm:", layer_norm_tokens([[1, 10], [3, 14]], [1, 1], [0, 0]))

# %% [markdown]
# | Метод | Група для статистик | Evaluation |
# |---|---|---|
# | BatchNorm, приклад `(B,D)` | стовпець: B прикладів | збережені running-оцінки за замовчуванням |
# | LayerNorm у Transformer `(B,T,D)` | D ознак одного токена | статистики поточного токена |
#
# Питання: чи зміниться LayerNorm першого токена, якщо додати padding-токени?
# А чи може padding вплинути на його attention без маски? Чому нормалізація
# не замінює attention mask?
#
# Сучасні моделі можуть використовувати RMSNorm — споріднену нормалізацію
# без віднімання середнього. У цій лабораторній реалізуємо саме LayerNorm.

# %% [markdown]
# ## 10. Dropout: випадкове вимкнення активацій
#
# Мережа може добре запам'ятати навчальні приклади, але помилятися на нових
# (overfitting). Dropout під час навчання додає шум: випадково обнуляє
# окремі активації. Це зменшує можливість покладатися на фіксований набір
# ознак і може покращити узагальнення. Занадто великий `p` заважає навчанню;
# покращення якості не гарантоване. Dropout не обчислює середнє чи дисперсію.
#
# Для кожної координати незалежно беремо $m_i\sim\mathrm{Bernoulli}(1-p)$:
# $$y_i=\frac{m_i x_i}{1-p}\quad\text{(training)},\qquad
# y_i=x_i\quad\text{(evaluation)}.$$
# Це **inverted dropout**. Наприклад, `p=0.5` та маска `[1,0,1,0]`
# перетворюють `[1,2,3,4]` на `[2,0,6,0]`. Масштаб зберігає
# $\mathbb E[y_i\mid x_i]=x_i$, але не кожен окремий вихід, його норму чи
# дисперсію. За фіксованого x дисперсія шуму дорівнює $x_i^2p/(1-p)$.
#
# На evaluation нічого не вимикаємо і додатково не масштабуємо.
# Це не означає, що вихід усієї нелінійної мережі точно дорівнює середньому
# виходу всіх її випадкових training-версій.
#
# Реалізуйте `dropout` для списку. Використайте `rng.random()` для кожного
# елемента: значення < p означає вимкнення. Переданий генератор дозволяє
# відтворювати експерименти. Вимагайте `0 <= p < 1`, інакше `ValueError`.
# Для evaluation або `p=0` поверніть копію без викликів генератора.
# Порожній список повертає `[]`. Входи не змінюйте.

# %%
def dropout(x, p, training, rng):
    ...


def test_dropout():
    from random import Random
    class FixedRandom:
        def __init__(self, values):
            self.values = iter(values)
        def random(self):
            return next(self.values)
    x = [1.0, 2.0, 3.0, 4.0]
    result = dropout(x, 0.5, True, FixedRandom([0.9, 0.1, 0.8, 0.2]))
    assert result == [2.0, 0.0, 6.0, 0.0]
    assert x == [1.0, 2.0, 3.0, 4.0]
    for p, training in [(0.5, False), (0.0, True)]:
        result = dropout(x, p, training, FixedRandom([]))
        assert result == x and result is not x
    assert dropout([], 0.5, True, FixedRandom([])) == []
    # Однаковий seed відтворює маску, не фіксує її на весь процес навчання.
    assert dropout(x, .3, True, Random(42)) == dropout(x, .3, True, Random(42))
    # Рівноймовірні маски одного елемента при p=.5: точне середнє.
    off = dropout([3.0], .5, True, FixedRandom([.1]))[0]
    on = dropout([3.0], .5, True, FixedRandom([.9]))[0]
    assert (off + on) / 2 == 3.0
    assert dropout([2.0], .25, True, FixedRandom([.25])) == [2.0 / .75]
    for p in [-0.1, 1.0, 2.0]:
        try:
            dropout(x, p, False, FixedRandom([]))
        except ValueError:
            pass
        else:
            raise AssertionError("Некоректна ймовірність")


if __name__ == "__main__":
    from random import Random
    test_dropout()
    rng = Random(7)  # створюємо ОДИН раз, поза training-циклом
    for step in range(3):
        print("train", step, dropout([1, 2, 3, 4], .5, True, rng))
    print("eval", dropout([1, 2, 3, 4], .5, False, rng))
    print("✓ inverted dropout")

# %% [markdown]
# **Backward:** для використаної у forward маски
# $\partial\mathcal L/\partial x_i=(\partial\mathcal L/\partial y_i)m_i/(1-p)$.
# Вимкнена координата має нульовий градієнт через цю операцію, збережена —
# масштабований. Для backward не можна згенерувати іншу маску.
# Autograd зберігає потрібну інформацію самостійно.
# У residual-блоці вимкнення гілки не перекриває окремий identity-шлях.
#
# Завдання для дослідження: для `[1,2,3,4]` усередніть виходи 10000 викликів
# із `p=.5`. Чи наближається кожна координата до входу? Потім порівняйте
# розкид із `p=.1` та `.9`. Не перевстановлюйте seed усередині циклу.

# %% [markdown]
# ## 11. LayerNorm і Dropout у залишкових гілках encoder
#
# Тепер попередній каркас отримує конкретний LayerNorm і Dropout:
# $$h=x+\mathrm{Dropout}(\mathrm{Attention}(\mathrm{LN}_1(x))),$$
# $$y=h+\mathrm{Dropout}(\mathrm{FFN}(\mathrm{LN}_2(h))).$$
# Обидві гілки повертають `d_model` ознак. FFN — знайома композиція
# `linear → GELU → linear` з розмірами `d_model → d_ff → d_model`;
# ті самі ваги застосовуються до кожного токена.
# Тут dropout діє на вихід гілки **перед residual-додаванням**. Shortcut
# залишається відкритим. Можливі також dropout на attention-вагах після
# softmax, на embedding-вході та всередині FFN; це окремі місця застосування.
# Після dropout attention-ваги не зобов'язані сумуватися до 1 у кожній масці.
#
# Нижче завершіть блок для одного вектора, викликаючи передані функції
# `norm1`, `attention`, `norm2`, `ffn` та наші `dropout`, `residual_add`.
# Справжній attention працюватиме з усією послідовністю; тестова функція
# тут потрібна лише для перевірки композиції. Генератор має бути спільним,
# але кожний виклик dropout повинен брати нові випадкові числа.

# %%
def encoder_with_dropout(x, attention, ffn, norm1, norm2, p, training, rng):
    ...


def test_encoder_with_dropout():
    from random import Random
    norm = lambda x: layer_norm(x, [1, 1], [0, 0])
    attention = lambda x: [2.0, 4.0]
    ffn = lambda x: [6.0, 8.0]
    x = [1.0, 3.0]
    expected = encoder_residuals(x, attention, ffn, norm, norm)
    assert encoder_with_dropout(x, attention, ffn, norm, norm, .5, False, Random(1)) == expected
    class FixedRandom:
        def __init__(self):
            self.values = iter([.9, .1, .1, .9])
        def random(self):
            return next(self.values)
    assert encoder_with_dropout(x, attention, ffn, norm, norm, .5, True, FixedRandom()) == [5.0, 19.0]
    zero = lambda x: [0.0, 0.0]
    assert encoder_with_dropout(x, zero, zero, norm, norm, .5, True, Random(2)) == x
    assert x == [1.0, 3.0]


if __name__ == "__main__":
    test_encoder_with_dropout()
    print("✓ pre-norm + dropout + residual")

# %% [markdown]
# У PyTorch відповідники — `nn.LayerNorm(d_model)` та `nn.Dropout(p)`.
# `model.train()` вмикає training-поведінку Dropout і BatchNorm;
# `model.eval()` перемикає її на evaluation. LayerNorm має ту саму формулу
# в обох режимах. `torch.no_grad()` вимикає запис графа похідних, але
# **не перемикає Dropout у evaluation**: для inference зазвичай потрібні обидва.
#
# Dropout застосовували в початковому Transformer (базове `p=.1`), проте
# його місце й імовірність залежать від архітектури та режиму навчання.
# Він є налаштовуваною регуляризацією; `p=0` цілком допустимий.
# Щоб перевірити користь, порівнюйте якість на відкладених даних: самі
# приклади масок вище не доводять покращення узагальнення.
#
# Джерела для цих розділів:
# [BatchNorm](https://arxiv.org/abs/1502.03167),
# [LayerNorm](https://arxiv.org/abs/1607.06450),
# [Dropout](https://jmlr.org/papers/v15/srivastava14a.html),
# [PyTorch BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html),
# [PyTorch LayerNorm](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html),
# [PyTorch Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html).

# %% [markdown]
# ## Що здати
#
# 1. Реалізації функцій і перевірки, що проходять.
# 2. Таблицю градієнтів за глибиною та пояснення зникнення / вибуху через
#    добуток похідних. Окремо поясніть випадок $F'(x)=-1$.
# 3. Таблицю наближення до $e$ та пояснення, чому потрібен масштаб $T/N$.
# 4. Формулу backward для residual-блока і схему двох гілок encoder.
# 5. Порівняння осей BatchNorm та LayerNorm на одній матриці; пояснення
#    training/evaluation, ролі epsilon та параметрів gamma/beta.
# 6. Реалізацію Dropout, результати усереднення масок та формулу його backward.
# 7. Каркас encoder із LayerNorm і Dropout; пояснення місця dropout у гілці.
#
# Джерела для поглиблення:
# [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027),
# [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (post-norm),
# [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
# (перехід від residual-кроків до неперервної динаміки).
