// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Physical AI and Humanoid Robots',
  tagline: 'A simulation-first, hands-on textbook',
  favicon: 'img/favicon.ico',

  // GitHub Pages config
  url: 'https://your-username.github.io',
  baseUrl: '/textbook-ai-robotics/',
  organizationName: 'your-username',
  projectName: 'textbook-ai-robotics',
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
          routeBasePath: '/', // 👈 Docs ARE the site
          sidebarPath: './sidebars.js',
          editUrl:
            'https://github.com/your-username/textbook-ai-robotics/edit/main/',
        },

        blog: false, // ❌ Blog disabled

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
      label: 'Book', // this is the main landing page
    },
    {
      href: 'https://github.com/your-username/textbook-ai-robotics',
      label: 'GitHub',
      position: 'right',
    },
  ],
},
    footer: undefined, // ❌ Remove footer completely

    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json', 'cpp'],
    },
  },
};

export default config;
