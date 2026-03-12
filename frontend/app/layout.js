import './globals.css';

export const metadata = {
  title: 'Heart Disease ML Studio',
  description: 'Interface statique pour exploration, entraînement et comparaison de modèles ML'
};

export default function RootLayout({ children }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
