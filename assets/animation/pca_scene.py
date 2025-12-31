from manim import *
import numpy as np
from sklearn.preprocessing import StandardScaler

class PCA2DScene(Scene):
    def construct(self):
        np.random.seed(42)
        X = np.random.randn(200, 2) + np.array([3, -2])
        
        plane = NumberPlane(x_range=[-14, 14, 1], y_range=[-5, 5, 1], background_line_style={"stroke_opacity": 0})
        self.add(plane)
        
        def points_to_mobjects(data, color=BLUE):
            return VGroup(*[Dot(point=[x, y, 0], radius=0.08, color=color) for x, y in data])
        
        dots_orig = points_to_mobjects(X)
        self.play(FadeIn(dots_orig))
        self.wait(1)
        
        step_text = Text("1. Center Data", font_size=30).to_corner(UL).shift(RIGHT*0.5 + DOWN*0.5)
        self.add(step_text)
        
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        dots_centered = points_to_mobjects(X_centered, color=GREEN)
        self.play(Transform(dots_orig, dots_centered))
        self.wait(1)
        self.play(FadeOut(step_text))
        
        step_text = Text("2. Eigenvectors", font_size=30).to_corner(UL).shift(RIGHT*0.5 + DOWN*0.5)
        self.add(step_text)
        
        cov = np.cov(X_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        
        eigvecs_lines = VGroup(*[
            Arrow(start=ORIGIN, end=[eigvecs[0,i]*3, eigvecs[1,i]*3, 0], buff=0, color=RED)
            for i in range(eigvecs.shape[1])
        ])
        self.play(FadeIn(eigvecs_lines))
        self.wait(1)
        self.play(FadeOut(step_text))

        step_text = Text("3. Principal Component", font_size=30).to_corner(UL).shift(RIGHT*0.5 + DOWN*0.5)
        self.add(step_text)
        
        top_vec = eigvecs[:, 0]
        top_arrow = Arrow(start=ORIGIN, end=[top_vec[0]*3, top_vec[1]*3, 0], buff=0, color=YELLOW)
        self.play(FadeIn(top_arrow))
        self.wait(1)
        self.play(FadeOut(step_text))

        step_text = Text("4. Project Data", font_size=30).to_corner(UL).shift(RIGHT*0.5 + DOWN*0.5)
        self.add(step_text)
        
        X_proj = X_centered @ top_vec[:, np.newaxis] @ top_vec[np.newaxis, :]
        dots_proj = points_to_mobjects(X_proj, color=ORANGE)
        self.play(Transform(dots_centered, dots_proj))
        self.wait(2)
        self.play(FadeOut(step_text))
