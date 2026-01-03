// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI and Humanoid Robots',
  tagline: 'A simulation-first, hands-on textbook',
  favicon: 'img/favicon.ico',

  // GitHub Pages config
  url: 'https://sufyanali92.github.io',
  baseUrl: '/Book_HAKATHON/',
  organizationName: 'sufyanali92', // GitHub username
  projectName: 'Book_HAKATHON', // GitHub repo name
  deploymentBranch: 'gh-pages',

  trailingSlash: false,

  onBrokenLinks: 'throw',
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
          routeBasePath: '/', // Docs are the main site
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/sufyanali92/Book_HAKATHON/edit/main/',
        },

        blog: false, // Blog disabled

        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: '', // optional: remove homepage title
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar', // your main sidebar
          position: 'left',
          label: 'Book', // main landing page
        },
        {
          href: 'https://github.com/sufyanali92/Book_HAKATHON',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: undefined, // Remove footer completely

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'cpp'],
    },
  },
};

export default config;
