import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# --- 物理常数 ---
VT = 0.02585  # 300K 时的热电压 (V)


def led_model_v(I, n, Is, Rs):
    """拟合方程：V = n*Vt*ln(I/Is + 1) + I*Rs"""
    # 限制 Is 不为 0 且 I/Is 为正
    return n * VT * np.log(np.maximum(I / Is, 1e-15) + 1) + I * Rs


class LEDFitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LED I-V 特性精准拟合器 (含 Rs 修正)")
        self.root.geometry("1100x750")

        # --- 布局 ---
        left_frame = ttk.Frame(root, padding="15")
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left_frame, text="1. 输入电流 I (mA) 列", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.i_text = tk.Text(left_frame, width=20, height=15)
        self.i_text.pack(pady=5)

        ttk.Label(left_frame, text="2. 输入电压 V (V) 列", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 0))
        self.v_text = tk.Text(left_frame, width=20, height=15)
        self.v_text.pack(pady=5)

        ttk.Button(left_frame, text="开始 LED 拟合", command=self.run_fit).pack(fill=tk.X, pady=15)
        ttk.Button(left_frame, text="清空数据",
                   command=lambda: [self.i_text.delete('1.0', tk.END), self.v_text.delete('1.0', tk.END)]).pack(
            fill=tk.X)

        self.plot_frame = ttk.Frame(root, padding="10")
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    def run_fit(self):
        try:
            # 1. 获取数据
            i_raw = np.array([float(x) for x in self.i_text.get('1.0', tk.END).split() if x.strip()])
            v_raw = np.array([float(x) for x in self.v_text.get('1.0', tk.END).split() if x.strip()])

            if len(i_raw) != len(v_raw) or len(i_raw) < 4:
                messagebox.showerror("数据错误", "请确保 I 和 V 行数一致且至少有 4 组数据")
                return

            # 2. 执行拟合 (注意：我们拟合的是 V = f(I)，这在数学上非常稳定)
            # 初始猜测：n=2.0, Is=1e-8 mA, Rs=10 Ohm
            popt, _ = curve_fit(
                led_model_v, i_raw, v_raw,
                p0=[2.0, 1e-8, 10.0],
                bounds=((0.1, 1e-12, 0), (50, 1, 5000))  # 限制参数范围
            )
            n_fit, Is_fit, Rs_fit = popt

            # 3. 计算误差并标红
            v_pred = led_model_v(i_raw, *popt)
            error = np.abs(v_raw - v_pred)
            std_err = np.std(error)
            outliers = error > (1.5 * std_err)

            # 4. 绘图
            for widget in self.plot_frame.winfo_children(): widget.destroy()
            fig, ax = plt.subplots(figsize=(6, 5))

            # 拟合曲线
            i_smooth = np.linspace(0.001, max(i_raw) * 1.1, 300)
            v_smooth = led_model_v(i_smooth, *popt)
            ax.plot(v_smooth, i_smooth, 'b-', label=f'LED拟合 (Rs={Rs_fit:.1f}Ω)')

            # 实验点
            ax.scatter(v_raw[~outliers], i_raw[~outliers], c='green', label='正常数据')
            ax.scatter(v_raw[outliers], i_raw[outliers], c='red', marker='x', s=80, label='误差较大')

            ax.set_xlabel("Voltage V (V)")
            ax.set_ylabel("Current I (mA)")
            ax.set_title("LED I-V Precision Fitting")
            ax.legend()
            ax.grid(True, alpha=0.3)

            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            NavigationToolbar2Tk(canvas, self.plot_frame).update()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # 显示结果
            res_str = f"拟合参数:\n理想因子 n: {n_fit:.2f}\n饱和电流 Is: {Is_fit:.2e} mA\n体电阻 Rs: {Rs_fit:.2f} Ω"
            ttk.Label(self.plot_frame, text=res_str, font=('Consolas', 10), foreground="blue").pack()

        except Exception as e:
            messagebox.showerror("失败", f"拟合错误：{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LEDFitApp(root)
    root.mainloop()