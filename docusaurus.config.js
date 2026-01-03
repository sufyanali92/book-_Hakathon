// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI and Humanoid Robots',
  tagline: 'A simulation-first, hands-on textbook',
  favicon: '/img/favicon.ico',

  // ✅ GitHub Pages deployment
  url: 'https://sufyanali92.github.io',
  baseUrl: '/Book_HAKATHON/',
  organizationName: 'sufyanali92',
  projectName: 'Book_HAKATHON',
  deploymentBranch: 'gh-pages',

  trailingSlash: false,

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'warn',
  onDuplicateRoutes: 'warn',

  // ✅ i18n (minimal & valid)
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  // ✅ FIXED: markdown hooks (v3.9+ correct)
  markdown: {
    format: 'mdx',
    mermaid: true,
    emoji: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
      onBrokenMarkdownImages: 'throw',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // Docs act as homepage
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/sufyanali92/Book_HAKATHON/edit/main/',
        },

        blog: false,

        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: '', // clean homepage
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Book',
        },
        {
          href: 'https://github.com/sufyanali92/Book_HAKATHON',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },

    // ✅ Proper way to remove footer
    footer: {
      style: 'dark',
      links: [],
      copyright: '',
    },

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'cpp'],
    },
  },
};

export default config;
