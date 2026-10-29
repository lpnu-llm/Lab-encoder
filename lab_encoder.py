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
# ## Що здати
#
# 1. Реалізації функцій і перевірки, що проходять.
# 2. Таблицю градієнтів за глибиною та пояснення зникнення / вибуху через
#    добуток похідних. Окремо поясніть випадок $F'(x)=-1$.
# 3. Таблицю наближення до $e$ та пояснення, чому потрібен масштаб $T/N$.
# 4. Формулу backward для residual-блока і схему двох гілок encoder.
#
# Джерела для поглиблення:
# [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027),
# [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (post-norm),
# [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
# (перехід від residual-кроків до неперервної динаміки).
