import './globals.css';

// Métadonnées statiques utilisées par Next.js pour l'en-tête du document.
export const metadata = {
  title: 'Heart Disease ML Studio',
  description: 'Interface statique pour exploration, entraînement et comparaison de modèles ML'
};

export default function RootLayout({ children }) {
  // Coquille de layout partagée pour toutes les routes de l'application.
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
