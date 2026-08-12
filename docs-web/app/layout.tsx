import type { Metadata } from "next";
import { JetBrains_Mono, Manrope, Syne } from "next/font/google";
import "./globals.css";

const syne = Syne({
  subsets: ["latin"],
  variable: "--font-syne",
  weight: ["500", "600", "700", "800"],
});

const manrope = Manrope({
  subsets: ["latin"],
  variable: "--font-manrope",
  weight: ["400", "500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains",
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "SynaptIX — Librería integral de Machine Learning para Python",
  description:
    "Aprendizaje supervisado, no supervisado, por refuerzo y redes neuronales con una API unificada en español. pip install synaptix",
  keywords: ["machine learning", "python", "synaptix", "deep learning", "IA"],
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="es">
      <body
        className={`${syne.variable} ${manrope.variable} ${jetbrains.variable} grain antialiased`}
      >
        {children}
      </body>
    </html>
  );
}
