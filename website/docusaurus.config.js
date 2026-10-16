// @ts-check

import {themes as prismThemes} from 'prism-react-renderer';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';

// The GitHub repository. Change these two if the repository has another owner or name (the site URL, the base path, and the links follow them)
const OWNER = 'Gucchi3';
const REPO  = 'hitokoto';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'hitokoto',
  tagline: 'A tiny language model that hears one phrase and answers with one phrase, on a microcontroller.',
  favicon: 'img/hitokoto-A-icon-light.svg',

  // GitHub Pages serves the site at https://OWNER.github.io/REPO/
  url: `https://${OWNER.toLowerCase()}.github.io`,
  baseUrl: `/${REPO}/`,
  organizationName: OWNER,
  projectName: REPO,
  trailingSlash: false,

  // Stop the build on a broken link, so CI finds it
  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  markdown: {
    format: 'detect',
    hooks: {
      onBrokenMarkdownLinks: 'throw',
      onBrokenMarkdownImages: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        respectPrefersColorScheme: true,
      },
      docs: {
        sidebar: {
          hideable: true,
        },
      },
      navbar: {
        logo: {
          alt: 'hitokoto',
          src: 'img/hitokoto-A-leaf-light.svg',
          srcDark: 'img/hitokoto-A-leaf-dark.svg',
        },
        items: [
          {type: 'docSidebar', sidebarId: 'docs', position: 'left', label: 'Docs'},
          {type: 'docSidebar', sidebarId: 'learn', position: 'left', label: 'Learn'},
          {type: 'docSidebar', sidebarId: 'api', position: 'left', label: 'API'},
          {href: `https://github.com/${OWNER}/${REPO}`, label: 'GitHub', position: 'right'},
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Docs',
            items: [
              {label: 'Get started', to: '/docs/intro'},
              {label: 'Roadmap', to: '/docs/roadmap'},
              {label: 'Learn', to: '/docs/learn'},
              {label: 'API', to: '/docs/api'},
            ],
          },
          {
            title: 'Links',
            items: [
              {label: 'GitHub', href: `https://github.com/${OWNER}/${REPO}`},
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Kohei Yamaguchi`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.vsDark,
        additionalLanguages: ['bash', 'toml'],
      },
    }),
};

export default config;
