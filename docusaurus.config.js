// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI and Humanoid Robots',
  tagline: 'A simulation-first, hands-on textbook',
  favicon: 'img/favicon.ico',

  // ✅ GitHub Pages config (CORRECT)
  url: 'https://sufyanali92.github.io',
  baseUrl: '/Book_HAKATHON/',
  organizationName: 'sufyanali92',
  projectName: 'Book_HAKATHON',
  deploymentBranch: 'gh-pages',

  trailingSlash: true,

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/', // ✅ Docs ARE the site
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/sufyanali92/Book_HAKATHON/edit/main/',
        },

        blog: false, // ❌ Blog disabled (book only)

        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: '', // clean book-style UI
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

    footer: undefined, // ❌ No footer (textbook style)

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'cpp'],
    },
  },
};

export default config;
