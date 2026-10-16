import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Heading from '@theme/Heading';
import Layout from '@theme/Layout';
import ThemedImage from '@theme/ThemedImage';

import styles from './index.module.css';

// The three points of hitokoto under the logo (all of them are still being built)
const FEATURES = [
  {
    title: 'One phrase in, one phrase out',
    text: 'hitokoto (一言, "a word") will answer a short phrase with a short phrase. It aims at simple daily talk, like a pet robot or a smart speaker.',
  },
  {
    title: 'Runs on tiny devices',
    text: 'The model will be small enough for a microcontroller such as the ESP32, or for a small FPGA. Its memory for the talk will stay the same size, however long you talk.',
  },
  {
    title: 'Train it yourself',
    text: 'One GPU will be enough to pretrain the model, teach it daily talk, and get it ready for your device.',
  },
];


export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  const logo         = {light: useBaseUrl('/img/hitokoto-A-leaf-light.svg'), dark: useBaseUrl('/img/hitokoto-A-leaf-dark.svg')};
  return (
    <Layout description={siteConfig.tagline}>
      <header className={styles.hero}>
        <div className="container">
          <Heading as="h1" className={styles.title}>
            <ThemedImage className={styles.logo} alt={siteConfig.title} sources={logo} />
          </Heading>
          <p className={styles.tagline}>{siteConfig.tagline}</p>
          <p><span className="coming-soon">Coming soon</span></p>
          <div className={styles.buttons}>
            <Link className="button button--primary button--lg" to="/docs/intro">Read the docs</Link>
            <Link className="button button--secondary button--outline button--lg" to="/docs/roadmap">See the roadmap</Link>
          </div>
        </div>
      </header>
      <main className="container">
        <div className={`row ${styles.features}`}>
          {FEATURES.map(({title, text}) => (
            <section key={title} className="col col--4">
              <Heading as="h2" className={styles.featureTitle}>{title}</Heading>
              <p>{text}</p>
            </section>
          ))}
        </div>
      </main>
    </Layout>
  );
}
