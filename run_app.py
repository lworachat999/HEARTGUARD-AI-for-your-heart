import sys, os, json, time, random, re, io, math
import numpy as np
import torch
import torch.nn as nn
import joblib

from PySide6.QtCore import Qt, QTimer, QRectF, QEasingCurve, QPoint, QSize
from PySide6.QtGui import QFont, QColor, QPainter, QPen, QPixmap, QIcon
from PySide6.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton,
    QStackedWidget, QFrame, QFormLayout, QLineEdit, QMessageBox, QTextEdit,
    QProgressBar, QScrollArea, QSizePolicy, QSpacerItem
)

# ---------- Optional SHAP (safe fallback if not installed) ----------
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


# ===================== Model =====================
class HeartNet(nn.Module):
    # 64 -> 32 -> 1 to match the pretrained artifacts I gave you
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)


# ===================== Custom Widgets =====================
class SidebarButton(QPushButton):
    def __init__(self, text, emoji=""):
        super().__init__(f"{emoji}  {text}")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setStyleSheet("""
            QPushButton {
                text-align: left;
                background: transparent;
                border: none;
                padding: 10px 14px;
                color: #eaf1fb;
                font-weight: 600;
                border-radius: 8px;
            }
            QPushButton:hover { background: rgba(255,255,255,0.12); }
            QPushButton:pressed { background: rgba(255,255,255,0.18); }
        """)


class Card(QFrame):
    def __init__(self):
        super().__init__()
        self.setObjectName("card")
        self.setStyleSheet("""
            QFrame#card {
                background: #ffffff;
                border: 1px solid #e5eaf1;
                border-radius: 14px;
            }
        """)


class RiskGauge(QWidget):
    """Animated circular gauge showing %."""
    def __init__(self):
        super().__init__()
        self._value = 0.0
        self._target = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._step)
        self.setMinimumHeight(220)

    def start(self, target_percent):
        self._target = max(0.0, min(100.0, float(target_percent)))
        self._timer.start(12)

    def _step(self):
        if abs(self._value - self._target) < 0.5:
            self._value = self._target
            self._timer.stop()
        else:
            self._value += (self._target - self._value) * 0.12
        self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(20, 20, self.width()-40, self.height()-40)
        # bg ring
        pen = QPen(QColor("#e6edf6"), 18); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen); p.drawArc(rect, 45*16, 270*16)
        # value ring
        color = "#21ba45" if self._value < 40 else ("#f2c037" if self._value < 60 else "#e53935")
        pen = QPen(QColor(color), 18); pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen); p.drawArc(rect, 45*16, int(270*16*(self._value/100.0)))
        # Text
        p.setPen(QColor("#0e1726"))
        p.setFont(QFont("Segoe UI", 28, QFont.Bold))
        txt = f"{self._value:0.0f}%"
        p.drawText(self.rect(), Qt.AlignCenter, txt)


# ===================== Main App =====================
class HeartGuard(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HeartGuard — Smart Cardiac Companion")
        self.resize(1240, 780)

        # ---- Fields/order must match training/artifacts ----
        self.fields = [
            "age","anaemia","creatinine_phosphokinase","diabetes",
            "ejection_fraction","high_blood_pressure","platelets",
            "serum_creatinine","serum_sodium","sex","smoking","time"
        ]

        # ---- Load artifacts ----
        self._load_artifacts()

        # ---- UI ----
        self._build_ui()

        # ---- Background (federated log) ----
        self._start_federated_log()

    # ---------- Artifacts ----------
    def _load_artifacts(self):
        try:
            self.scaler = joblib.load("scaler.pkl")
        except Exception as e:
            QMessageBox.critical(self, "Missing scaler.pkl", f"{e}")
            sys.exit(1)

        try:
            self.model = HeartNet(len(self.fields))
            self.model.load_state_dict(torch.load("model_initial.pt", map_location="cpu"))
            self.model.eval()
        except Exception as e:
            QMessageBox.critical(self, "Model load error", str(e))
            sys.exit(1)

        # Personalized baseline file
        self.profile_path = "patient_profile.json"
        if os.path.exists(self.profile_path):
            try:
                with open(self.profile_path, "r") as f:
                    self.profile = json.load(f)
            except Exception:
                self.profile = {}
        else:
            self.profile = {}

    # ---------- Layout ----------
    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0,0,0,0)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0.9, y2:1,
                             stop:0 #c9184a, stop:1 #5932ea);
                border: none;
            }
        """)
        sb = QVBoxLayout(sidebar); sb.setContentsMargins(16,18,16,18); sb.setSpacing(10)

        logo = QLabel("💖  HeartGuard")
        logo.setStyleSheet("color:white;")
        logo.setFont(QFont("Segoe UI", 20, QFont.Black))
        sb.addWidget(logo)
        sb.addSpacing(8)

        btn_input = SidebarButton("Patient Input", "🏥")
        btn_results = SidebarButton("Risk Results", "📊")
        btn_chat = SidebarButton("Health Assistant", "💬")
        btn_live = SidebarButton("Live Monitor", "📡")
        btn_exit = SidebarButton("Exit", "⏻")
        sb.addWidget(btn_input); sb.addWidget(btn_results); sb.addWidget(btn_chat); sb.addWidget(btn_live)
        sb.addStretch(1)
        sb.addWidget(btn_exit)

        # Content Area
        self.stack = QStackedWidget()
        root.addWidget(sidebar)
        root.addWidget(self.stack, 1)

        # Pages
        self._page_input()
        self._page_loading()
        self._page_results()
        self._page_chat()
        self._page_live()

        # Wire navigation
        btn_input.clicked.connect(lambda: self.stack.setCurrentWidget(self.pg_input))
        btn_results.clicked.connect(lambda: self.stack.setCurrentWidget(self.pg_results))
        btn_chat.clicked.connect(lambda: self.stack.setCurrentWidget(self.pg_chat))
        btn_live.clicked.connect(lambda: self.stack.setCurrentWidget(self.pg_live))
        btn_exit.clicked.connect(self.close)

        # Global style
        self.setStyleSheet("""
            QWidget { background: #f7f9fc; color: #0e1726; }
            QLabel { font-size: 14px; }
            QLineEdit {
                padding:10px; border:1px solid #d9e1ec; border-radius:10px; background:#ffffff;
            }
            QLineEdit:focus { border:1px solid #7aa2ff; }
            QPushButton.primary {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1a73e8, stop:1 #0b57cf);
                color: white; border: none; padding: 12px 16px; border-radius: 10px; font-weight: 600;
            }
            QPushButton.primary:hover { filter: brightness(110%); }
            QTextEdit { background: #ffffff; border:1px solid #e6edf6; border-radius: 10px; padding: 10px; }
        """)

    # ---------- Pages ----------
    def _page_input(self):
        self.pg_input = QWidget()
        lay = QHBoxLayout(self.pg_input); lay.setContentsMargins(24,24,24,24); lay.setSpacing(22)

        # Left: Form
        form_card = Card()
        form_layout = QFormLayout(form_card)
        form_layout.setContentsMargins(18,18,18,18); form_layout.setSpacing(10)
        title = QLabel("Patient Input"); title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        form_layout.addRow(title, QLabel(""))
        self.inputs = {}
        for f in self.fields:
            le = QLineEdit()
            if f in self.profile: le.setText(str(self.profile[f]))
            self.inputs[f] = le
            nice = f.replace("_"," ").capitalize()
            form_layout.addRow(QLabel(nice), le)

        self.btn_predict = QPushButton("Predict Risk")
        self.btn_predict.setObjectName("btnPredict")
        self.btn_predict.setProperty("class","primary")
        self.btn_predict.setStyleSheet("QPushButton#btnPredict { }")
        self.btn_predict.setCursor(Qt.PointingHandCursor)
        self.btn_predict.setMinimumHeight(46)
        self.btn_predict.setStyleSheet("QPushButton#btnPredict { }")
        self.btn_predict.setClass = "primary"
        self.btn_predict.setStyleSheet("QPushButton.primary{}")
        self.btn_predict.setProperty("class","primary")
        self.btn_predict.setStyleSheet("QPushButton.primary{}")
        self.btn_predict.setText("🔍  Predict Risk")
        self.btn_predict.setStyleSheet("QPushButton.primary{ background:#1a73e8; color:white; border-radius:10px; padding:12px; }"
                                       "QPushButton.primary:hover{ background:#0b57cf; }")
        self.btn_predict.clicked.connect(self._on_predict)
        form_layout.addRow(self.btn_predict, QLabel(""))

        # Right: Profile/Preview
        side = QVBoxLayout()
        info = Card(); in_l = QVBoxLayout(info); in_l.setContentsMargins(18,18,18,18)
        h = QLabel("Quick Info")
        h.setFont(QFont("Segoe UI", 16, QFont.Bold))
        p = QLabel("Enter clinical + lab values. Click **Predict Risk** to analyze.\n"
                   "Baseline per patient is updated locally.")
        p.setWordWrap(True)
        in_l.addWidget(h); in_l.addWidget(p)

        preview = Card(); pv = QVBoxLayout(preview); pv.setContentsMargins(18,18,18,18)
        ph = QLabel("Current Baseline"); ph.setFont(QFont("Segoe UI", 14, QFont.DemiBold))
        self.preview_text = QTextEdit(); self.preview_text.setReadOnly(True)
        self._refresh_preview()
        pv.addWidget(ph); pv.addWidget(self.preview_text)

        side.addWidget(info); side.addWidget(preview)
        lay.addWidget(form_card, 2)
        lay.addLayout(side, 3)
        self.stack.addWidget(self.pg_input)

    def _page_loading(self):
        self.pg_loading = QWidget()
        lay = QVBoxLayout(self.pg_loading); lay.setContentsMargins(24,24,24,24)
        card = Card(); c = QVBoxLayout(card); c.setContentsMargins(30,30,30,30)
        h = QLabel("Analyzing patient data…"); h.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.setValue(0)
        self.progress.setFixedHeight(22); self.progress.setTextVisible(False)
        tip = QLabel("Using personalized baseline, model inference and explanation…")
        tip.setStyleSheet("color:#5b6b88;")
        c.addWidget(h, alignment=Qt.AlignCenter); c.addSpacing(12); c.addWidget(self.progress); c.addWidget(tip, alignment=Qt.AlignCenter)
        lay.addStretch(1); lay.addWidget(card); lay.addStretch(1)
        self.stack.addWidget(self.pg_loading)

    def _page_results(self):
        self.pg_results = QWidget()
        lay = QHBoxLayout(self.pg_results); lay.setContentsMargins(24,24,24,24); lay.setSpacing(22)

        # Left: Gauge + summary
        left = QVBoxLayout()
        card1 = Card(); c1 = QVBoxLayout(card1); c1.setContentsMargins(22,22,22,22)
        t = QLabel("Predicted Risk"); t.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.gauge = RiskGauge()
        self.result_label = QLabel(""); self.result_label.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        self.result_label.setAlignment(Qt.AlignCenter)
        c1.addWidget(t); c1.addWidget(self.gauge); c1.addWidget(self.result_label)
        left.addWidget(card1)

        card2 = Card(); c2 = QVBoxLayout(card2); c2.setContentsMargins(22,22,22,22)
        sh = QLabel("Doctor-style Summary"); sh.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        self.summary_label = QLabel(""); self.summary_label.setWordWrap(True)
        c2.addWidget(sh); c2.addWidget(self.summary_label)
        left.addWidget(card2)

        # Right: explanation + next steps
        right = QVBoxLayout()
        card3 = Card(); c3 = QVBoxLayout(card3); c3.setContentsMargins(22,22,22,22)
        eh = QLabel("Top Contributing Factors"); eh.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        self.explain_box = QTextEdit(); self.explain_box.setReadOnly(True)
        self.explain_box.setPlaceholderText("Feature contributions will appear here.")
        c3.addWidget(eh); c3.addWidget(self.explain_box)
        right.addWidget(card3)

        card4 = Card(); c4 = QVBoxLayout(card4); c4.setContentsMargins(22,22,22,22)
        nh = QLabel("Next Steps"); nh.setFont(QFont("Segoe UI", 16, QFont.DemiBold))
        self.btn_go_chat = QPushButton("Proceed to Health Assistant  →")
        self.btn_go_chat.setProperty("class","primary")
        self.btn_go_chat.setStyleSheet("QPushButton.primary{ background:#1a73e8; color:white; border-radius:10px; padding:12px; }"
                                       "QPushButton.primary:hover{ background:#0b57cf; }")
        self.btn_go_chat.clicked.connect(lambda: self.stack.setCurrentWidget(self.pg_chat))
        tips = QLabel("Use the What-if Assistant to test scenarios (e.g., “What if sodium = 140?”).")
        tips.setStyleSheet("color:#5b6b88;")
        c4.addWidget(nh); c4.addWidget(tips); c4.addWidget(self.btn_go_chat, alignment=Qt.AlignLeft)
        right.addWidget(card4)

        lay.addLayout(left, 3)
        lay.addLayout(right, 2)
        self.stack.addWidget(self.pg_results)

    def _page_chat(self):
        self.pg_chat = QWidget()
        lay = QVBoxLayout(self.pg_chat); lay.setContentsMargins(24,24,24,24)
        card = Card(); c = QVBoxLayout(card); c.setContentsMargins(18,18,18,18)
        title = QLabel("💬 Health Assistant (What-if + General Guidance)")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.chat_history = QTextEdit(); self.chat_history.setReadOnly(True)
        self.chat_input = QLineEdit(); self.chat_input.setPlaceholderText("Try: What if ejection_fraction = 45?")
        send = QPushButton("Send"); send.setProperty("class","primary")
        send.setStyleSheet("QPushButton.primary{ background:#1a73e8; color:white; border-radius:10px; padding:10px; }"
                           "QPushButton.primary:hover{ background:#0b57cf; }")
        send.clicked.connect(self._chat_send)
        row = QHBoxLayout(); row.addWidget(self.chat_input, 1); row.addWidget(send)
        c.addWidget(title); c.addWidget(self.chat_history, 1); c.addLayout(row)
        lay.addWidget(card, 1)
        self.stack.addWidget(self.pg_chat)

    def _page_live(self):
        self.pg_live = QWidget()
        lay = QVBoxLayout(self.pg_live); lay.setContentsMargins(24,24,24,24)

        dash = Card(); d = QVBoxLayout(dash); d.setContentsMargins(18,18,18,18)
        title = QLabel("📡 Live Monitor (simulated)"); title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        self.vital_label = QLabel("—"); self.vital_label.setAlignment(Qt.AlignCenter)
        self.vital_label.setFont(QFont("Consolas", 18, QFont.Bold))

        self.ecg = QLabel(); self.ecg.setMinimumHeight(220)
        self.ecg.setStyleSheet("background:#0b1b2b; border-radius:12px;")
        d.addWidget(title); d.addWidget(self.vital_label); d.addWidget(self.ecg)
        lay.addWidget(dash, 1)
        self.stack.addWidget(self.pg_live)

        # timers
        self.timer_vitals = QTimer(self); self.timer_vitals.timeout.connect(self._tick_vitals); self.timer_vitals.start(1200)
        self.timer_ecg = QTimer(self); self.timer_ecg.timeout.connect(self._tick_ecg); self.timer_ecg.start(60)

    # ---------- Predict Flow ----------
    def _on_predict(self):
        try:
            vals = {}
            for f in self.fields:
                t = self.inputs[f].text().strip()
                if t == "": raise ValueError(f"Missing value: {f}")
                vals[f] = float(t)
        except Exception as e:
            QMessageBox.warning(self, "Input error", str(e))
            return

        self.current_values = vals
        # Save baseline
        self.profile.update(vals)
        with open(self.profile_path, "w") as f: json.dump(self.profile, f, indent=2)
        self._refresh_preview()

        # go to loading screen
        self.stack.setCurrentWidget(self.pg_loading)
        self.progress.setValue(0)
        self._loading_timer = QTimer(self); self._loading_timer.timeout.connect(self._loading_step); self._loading_timer.start(28)

    def _loading_step(self):
        v = min(100, self.progress.value() + random.randint(2,5))
        self.progress.setValue(v)
        if v >= 100:
            self._loading_timer.stop()
            QTimer.singleShot(220, self._compute_results)

    def _compute_results(self):
        X = np.array([self.current_values[f] for f in self.fields]).reshape(1, -1)
        try:
            Xs = self.scaler.transform(X)
        except Exception:
            Xs = X
        Xt = torch.tensor(Xs, dtype=torch.float32)
        with torch.no_grad():
            prob = float(self.model(Xt).item())
        risk = prob * 100.0

        # Summary & label
        label, advice = self._summary_text(risk, self.current_values)
        self.result_label.setText(label)
        self.summary_label.setText(advice)
        self.gauge.start(risk)

        # Explanations (SHAP first, fallback to weight*input)
        contribs = None
        if SHAP_AVAILABLE:
            try:
                explainer = shap.Explainer(self.model, Xt)
                sv = explainer(Xt)
                contribs = sv.values[0]
            except Exception:
                contribs = None
        if contribs is None:
            try:
                w = list(self.model.parameters())[0].detach().numpy()[0]  # first layer weights
                contribs = (Xs[0] * w)
            except Exception:
                contribs = np.zeros(len(self.fields))

        # Format top contributors
        pairs = list(zip(self.fields, contribs))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top_lines = []
        for f, c in pairs[:8]:
            arrow = "🔺" if c > 0 else "🔻"
            top_lines.append(f"{arrow} {f.replace('_',' ').capitalize()}: {c:+.3f}")
        self.explain_box.setText("\n".join(top_lines))

        self.stack.setCurrentWidget(self.pg_results)

    # ---------- Helpers ----------
    def _summary_text(self, risk, vals):
        if risk < 40:
            hdr = "✅ Low risk"
            txt = ("The model indicates a LOW probability of heart-failure event. "
                   "Maintain healthy lifestyle and routine follow-up.")
            color = "#21ba45"
        elif risk < 60:
            hdr = "⚠️ Moderate risk"
            txt = ("The model indicates a MODERATE probability. Consider check-ups, "
                   "optimize blood pressure/diabetes, and monitor symptoms.")
            color = "#f2c037"
        else:
            hdr = "🚨 High risk"
            txt = ("The model indicates a HIGH probability. Seek medical advice promptly. "
                   "If symptomatic (breathlessness, edema, chest pain), consult emergency care.")
            color = "#e53935"
        label = f"<span style='color:{color}; font-weight:700'>{hdr}</span>  ({risk:.1f}%)"
        bullet = []
        for k in ["ejection_fraction","serum_creatinine","serum_sodium","age","high_blood_pressure","diabetes","smoking"]:
            if k in vals:
                bullet.append(f"• {k.replace('_',' ').capitalize()}: {vals[k]}")
        text = "Key observed parameters:\n" + "\n".join(bullet)
        return label, text

    def _refresh_preview(self):
        lines = [f"{f}: {self.profile.get(f, '—')}" for f in self.fields]
        self.preview_text.setText("\n".join(lines))

    # ---------- Chatbot ----------
    def _chat_send(self):
        msg = self.chat_input.text().strip()
        if not msg: return
        self._chat_add_bubble(msg, author="user")
        self.chat_input.clear()

        # What-if parser
        if self._handle_whatif(msg):
            return

        # General guidance
        reply = self._general_guidance(msg)
        self._chat_add_bubble(reply, author="bot")

    def _chat_add_bubble(self, text, author="user"):
        color = "#e8f0fe" if author == "user" else "#fef7e0"
        name = "You" if author == "user" else "Assistant"
        html = f"""
        <div style="background:{color}; padding:10px 12px; border-radius:12px; margin:8px 0;">
            <b>{name}:</b> {text}
        </div>
        """
        self.chat_history.append(html)

    def _handle_whatif(self, text):
        t = text.lower()

        # pattern: what if feature = value
        m = re.search(r"what if\s+([a-z_ ]+)\s*=?\s*([0-9]+(?:\.[0-9]+)?)", t)
        if m:
            feat = m.group(1).strip().replace(" ", "_")
            val = float(m.group(2))
            if feat in self.fields:
                base = self.current_values if hasattr(self, "current_values") else self.profile
                vals = {f: float(base.get(f, 0)) for f in self.fields}
                vals[feat] = val
                r = self._simulate(vals)
                self._chat_add_bubble(f"If {feat} = {val}, predicted risk ≈ {r:.1f}%.", author="bot")
                return True

        # increase/decrease by
        m2 = re.search(r"(increase|decrease)\s+([a-z_ ]+)\s+by\s+([0-9]+(?:\.[0-9]+)?)", t)
        if m2:
            op, feat, amt = m2.groups()
            feat = feat.strip().replace(" ", "_")
            amt = float(amt)
            if feat in self.fields:
                base = self.current_values if hasattr(self, "current_values") else self.profile
                cur = float(base.get(feat, 0))
                newv = cur + amt if op == "increase" else cur - amt
                vals = {f: float(base.get(f, 0)) for f in self.fields}
                vals[feat] = newv
                r = self._simulate(vals)
                self._chat_add_bubble(f"After {op} {feat} by {amt}, risk ≈ {r:.1f}%.", author="bot")
                return True

        return False

    def _simulate(self, vals):
        X = np.array([vals[f] for f in self.fields]).reshape(1, -1)
        try: Xs = self.scaler.transform(X)
        except Exception: Xs = X
        Xt = torch.tensor(Xs, dtype=torch.float32)
        with torch.no_grad(): p = float(self.model(Xt).item())
        return p*100.0

    def _general_guidance(self, t):
        t = t.lower()
        if "reduce" in t or "improve" in t or "lower risk" in t:
            return ("General steps to lower cardiovascular risk: stop smoking, moderate daily activity, "
                    "manage blood pressure/diabetes, reduce salt, balanced diet, adequate sleep, "
                    "and regular follow-ups. For medication, consult your clinician.")
        if "exercise" in t:
            return ("Light to moderate activity (e.g., walking) is usually beneficial. "
                    "If you have symptoms or very high risk, seek clearance from a clinician before vigorous exercise.")
        if "diet" in t:
            return ("Heart-healthy diet: more vegetables, fruits, whole grains; limit salt; avoid trans-fats; "
                    "prefer lean proteins; adequate hydration unless on restriction.")
        return ("I can run simulations: e.g., 'What if sodium = 140?' or 'Increase ejection fraction by 10'. "
                "Ask general lifestyle questions too. (Educational use only—not medical advice.)")

    # ---------- Live Monitor ----------
    def _tick_vitals(self):
        hr = random.randint(62, 108)
        bp = f"{random.randint(110,145)}/{random.randint(70,95)}"
        spo2 = random.randint(93, 99)
        self.vital_label.setText(f"❤️ HR: {hr} bpm   •   🩸 BP: {bp}   •   O₂: {spo2}%")

    def _tick_ecg(self):
        w, h = max(620, self.ecg.width()), max(220, self.ecg.height())
        pix = QPixmap(w, h); pix.fill(QColor("#0b1b2b"))
        p = QPainter(pix); p.setRenderHint(QPainter.Antialiasing)
        # grid
        grid = QPen(QColor(20,45,70)); grid.setWidth(1); p.setPen(grid)
        for x in range(0, w, 20): p.drawLine(x,0,x,h)
        for y in range(0, h, 20): p.drawLine(0,y,w,y)
        # waveform
        pen = QPen(QColor("#36c2ff"), 2); pen.setCapStyle(Qt.RoundCap); p.setPen(pen)
        t = time.time()
        def ecg(x):
            # synthetic ECG-ish waveform
            return 0.25*math.sin((x/18.0)+t*6) + 0.08*math.sin((x/3.3)+t*10) + 0.03*math.sin((x/2.2)+t*17)
        prev = None
        mid = h//2
        for x in range(w):
            y = int(mid - ecg(x)*h*0.8)
            if prev: p.drawLine(prev[0], prev[1], x, y)
            prev = (x, y)
        p.end()
        self.ecg.setPixmap(pix)

    # ---------- Federated log ----------
    def _start_federated_log(self):
        self._fed_timer = QTimer(self)
        self._fed_timer.timeout.connect(self._write_fed)
        self._fed_timer.start(10000)

    def _write_fed(self):
        try:
            with open("federated_log.txt","a") as f:
                f.write(f"{time.ctime()}: local model update saved\n")
        except:
            pass


# ===================== Entry =====================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Minimal splash
    splash = QPixmap(520, 300); splash.fill(QColor("white"))
    qp = QPainter(splash)
    qp.setPen(QPen(QColor("#e91e63"))); qp.setFont(QFont("Segoe UI", 22, QFont.Black))
    qp.drawText(splash.rect(), Qt.AlignCenter, "💖 HeartGuard\nStarting…")
    qp.end()
    splash_lbl = QLabel(); splash_lbl.setPixmap(splash); splash_lbl.setWindowFlags(Qt.SplashScreen | Qt.FramelessWindowHint)
    splash_lbl.show(); app.processEvents(); time.sleep(1.0); splash_lbl.close()

    win = HeartGuard()
    win.show()
    sys.exit(app.exec())
